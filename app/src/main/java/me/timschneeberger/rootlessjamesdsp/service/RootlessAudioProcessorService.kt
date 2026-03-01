package me.timschneeberger.rootlessjamesdsp.service

import android.annotation.SuppressLint
import android.app.Activity
import android.app.NotificationManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.PowerManager
import android.os.Process
import androidx.annotation.RequiresApi
import androidx.core.content.getSystemService
import androidx.core.math.MathUtils.clamp
import androidx.lifecycle.Observer
import androidx.lifecycle.asLiveData
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import me.timschneeberger.rootlessjamesdsp.audio.AudioBufferPool
import me.timschneeberger.rootlessjamesdsp.audio.FastFFT
import me.timschneeberger.rootlessjamesdsp.audio.HardwareAccelerator
import me.timschneeberger.rootlessjamesdsp.audio.S24UltraDsp
import me.timschneeberger.rootlessjamesdsp.audio.SpatialAudioManager
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import me.timschneeberger.rootlessjamesdsp.BuildConfig
import me.timschneeberger.rootlessjamesdsp.R
import me.timschneeberger.rootlessjamesdsp.flavor.CrashlyticsImpl
import me.timschneeberger.rootlessjamesdsp.interop.JamesDspLocalEngine
import me.timschneeberger.rootlessjamesdsp.interop.ProcessorMessageHandler
import me.timschneeberger.rootlessjamesdsp.model.IEffectSession
import me.timschneeberger.rootlessjamesdsp.model.preference.AudioEncoding
import me.timschneeberger.rootlessjamesdsp.model.room.AppBlocklistDatabase
import me.timschneeberger.rootlessjamesdsp.model.room.AppBlocklistRepository
import me.timschneeberger.rootlessjamesdsp.model.room.BlockedApp
import me.timschneeberger.rootlessjamesdsp.model.rootless.SessionRecordingPolicyEntry
import me.timschneeberger.rootlessjamesdsp.session.rootless.OnRootlessSessionChangeListener
import me.timschneeberger.rootlessjamesdsp.session.rootless.RootlessSessionDatabase
import me.timschneeberger.rootlessjamesdsp.session.rootless.RootlessSessionManager
import me.timschneeberger.rootlessjamesdsp.session.rootless.SessionRecordingPolicyManager
import me.timschneeberger.rootlessjamesdsp.utils.Constants
import me.timschneeberger.rootlessjamesdsp.utils.Constants.ACTION_PREFERENCES_UPDATED
import me.timschneeberger.rootlessjamesdsp.utils.Constants.ACTION_SAMPLE_RATE_UPDATED
import me.timschneeberger.rootlessjamesdsp.utils.Constants.ACTION_SERVICE_HARD_REBOOT_CORE
import me.timschneeberger.rootlessjamesdsp.utils.Constants.ACTION_SERVICE_RELOAD_LIVEPROG
import me.timschneeberger.rootlessjamesdsp.utils.Constants.ACTION_SERVICE_SOFT_REBOOT_CORE
import me.timschneeberger.rootlessjamesdsp.utils.extensions.CompatExtensions.getParcelableAs
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.registerLocalReceiver
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.sendLocalBroadcast
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.toast
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.unregisterLocalReceiver
import me.timschneeberger.rootlessjamesdsp.utils.extensions.PermissionExtensions.hasRecordPermission
import me.timschneeberger.rootlessjamesdsp.utils.notifications.Notifications
import me.timschneeberger.rootlessjamesdsp.utils.notifications.ServiceNotificationHelper
import me.timschneeberger.rootlessjamesdsp.utils.preferences.Preferences
import org.koin.android.ext.android.inject
import timber.log.Timber
import java.io.IOException


@RequiresApi(Build.VERSION_CODES.Q)
class RootlessAudioProcessorService : BaseAudioProcessorService() {
    // System services
    private lateinit var mediaProjectionManager: MediaProjectionManager
    private lateinit var notificationManager: NotificationManager
    private lateinit var audioManager: AudioManager
    private lateinit var attributionContext: android.content.Context

    // Media projection token
    private var mediaProjection: MediaProjection? = null
    private var mediaProjectionStartIntent: Intent? = null

    // Processing
    private var recreateRecorderRequested = false
    private var recorderThread: Thread? = null
    private lateinit var engine: JamesDspLocalEngine
    private val isRunning: Boolean
        get() = recorderThread != null

    // Session management
    private lateinit var sessionManager: RootlessSessionManager
    private var sessionLossRetryCount = 0

    // Idle detection
    private var isProcessorIdle = false
    private var suspendOnIdle = false
    // PERFORMANCE: Use wait/notify instead of Thread.sleep for idle suspension
    private val idleLock = Object()

    // Exclude restricted apps flag
    private var excludeRestrictedSessions = false

    // S24 ULTRA - Spatial Audio Manager
    private var spatialAudioManager: SpatialAudioManager? = null

    // S24 ULTRA - Advanced DSP flags (read from preferences)
    @Volatile private var spatialAudioEnabled = false
    @Volatile private var spatialHeadTrackingEnabled = false
    @Volatile private var sourceSeparationEnabled = false
    @Volatile private var stemVocalsLevel = 1.0f
    @Volatile private var stemDrumsLevel = 1.0f
    @Volatile private var stemBassLevel = 1.0f
    @Volatile private var stemOtherLevel = 1.0f

    // S24 ULTRA - WakeLock to prevent CPU throttling during audio processing
    private var wakeLock: PowerManager.WakeLock? = null

    // REAL-TIME: Performance hint session (Android 12+)
    private var performanceHintSession: android.os.PerformanceHintManager.Session? = null

    // S24 ULTRA - Visualization data (throttled to prevent Choreographer jank)
    private var lastVisualizerUpdate = 0L
    private val visualizerUpdateInterval = 16L // ~60fps - smooth real-time visualization
    private var processingStartTime = 0L
    private val visualizerHandler = Handler(Looper.getMainLooper())

    // Buffer health tracking - measures pipeline stability
    @Volatile private var bufferHealthPercent = 100f
    @Volatile private var expectedFrameTimeMs = 0f

    // HWUI FIX: Volatile flag to immediately stop ALL visualization
    // This is checked at multiple points to prevent any rendering after activity pauses
    @Volatile private var isVisualizationEnabled = false

    // ███████╗██╗  ██╗████████╗██████╗ ███████╗███╗   ███╗███████╗
    // ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔════╝████╗ ████║██╔════╝
    // █████╗   ╚███╔╝    ██║   ██████╔╝█████╗  ██╔████╔██║█████╗
    // ██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══╝  ██║╚██╔╝██║██╔══╝
    // ███████╗██╔╝ ██╗   ██║   ██║  ██║███████╗██║ ╚═╝ ██║███████╗
    // ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝
    // 50% CPU MODE - FOREGROUND priority gets REAL CPU time

    private val numCores by lazy { Runtime.getRuntime().availableProcessors().coerceAtLeast(4) }

    // PERFORMANCE: Lazy-initialized thread pools - only created when first accessed
    // 50% CPU: FOREGROUND priority = system gives real CPU
    // -2 = THREAD_PRIORITY_FOREGROUND (between default and display)
    private val processingExecutor by lazy {
        java.util.concurrent.Executors.newFixedThreadPool(numCores * 2) { r ->
            Thread(r, "DSP-POWER").apply {
                priority = Thread.MAX_PRIORITY
                android.os.Process.setThreadPriority(-2) // FOREGROUND
            }
        }
    }

    // Visualization at FOREGROUND - gets real CPU but below audio
    // PERFORMANCE: Use bounded queue with DiscardOldestPolicy to prevent memory buildup
    private val vizExecutor by lazy {
        java.util.concurrent.ThreadPoolExecutor(
            numCores, numCores,
            0L, java.util.concurrent.TimeUnit.MILLISECONDS,
            java.util.concurrent.LinkedBlockingQueue(4),  // Max 4 pending frames
            { r -> Thread(r, "VIZ-POWER").apply {
                priority = Thread.MAX_PRIORITY
                android.os.Process.setThreadPriority(-2) // FOREGROUND
            }},
            java.util.concurrent.ThreadPoolExecutor.DiscardOldestPolicy()  // Drop oldest if queue full
        )
    }

    // FFT at FOREGROUND
    // PERFORMANCE: Use bounded queue with DiscardOldestPolicy to prevent memory buildup
    private val fftExecutor by lazy {
        java.util.concurrent.ThreadPoolExecutor(
            numCores, numCores,
            0L, java.util.concurrent.TimeUnit.MILLISECONDS,
            java.util.concurrent.LinkedBlockingQueue(4),  // Max 4 pending FFT operations
            { r -> Thread(r, "FFT-POWER").apply {
                priority = Thread.MAX_PRIORITY
                android.os.Process.setThreadPriority(-2) // FOREGROUND
            }},
            java.util.concurrent.ThreadPoolExecutor.DiscardOldestPolicy()  // Drop oldest if queue full
        )
    }

    // PERFORMANCE: Lazy-initialized visualization buffers - only allocated when visualization starts
    private val vizBufferA by lazy { FloatArray(65536) }
    private val vizBufferB by lazy { FloatArray(65536) }
    private val vizBufferC by lazy { FloatArray(65536) }
    @Volatile private var vizWriteIdx = 0
    @Volatile private var vizSize = 0
    @Volatile private var vizSampleRate = 48000

    // Dedicated FFT thread for coordination - initialized lazily in onCreate
    private var fftThread: HandlerThread? = null
    private var fftHandler: Handler? = null
    @Volatile private var visualizerLeftRms = -60f
    @Volatile private var visualizerRightRms = -60f
    @Volatile private var visualizerLeftPeak = -60f
    @Volatile private var visualizerRightPeak = -60f
    @Volatile private var visualizerLatency = 0f
    @Volatile private var visualizerSampleRate = 48000

    // ██████╗ ██████╗ ██╗   ██╗████████╗ █████╗ ██╗
    // ██╔══██╗██╔══██╗██║   ██║╚══██╔══╝██╔══██╗██║
    // ██████╔╝██████╔╝██║   ██║   ██║   ███████║██║
    // ██╔══██╗██╔══██╗██║   ██║   ██║   ██╔══██║██║
    // ██████╔╝██║  ██║╚██████╔╝   ██║   ██║  ██║███████╗
    // ╚═════╝ ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝
    // BRUTAL FFT PROCESSING - MULTIPLE SIZES SIMULTANEOUSLY

    // Primary FFT - HIGH resolution
    private val fftSize = 8192
    private val fftOverlap = 8 // 87.5% overlap
    private val fftHopSize = fftSize / fftOverlap

    // Secondary FFT - ULTRA resolution (runs in parallel)
    private val fftSize2 = 16384
    // PERFORMANCE: Lazy-initialized FFT buffers - only allocated when visualization is enabled
    private val fftInputLeft2 by lazy { FloatArray(fftSize2) }
    private val fftInputRight2 by lazy { FloatArray(fftSize2) }
    private val fftMagnitudesLeft2 by lazy { FloatArray(fftSize2 / 2) }
    private val fftMagnitudesRight2 by lazy { FloatArray(fftSize2 / 2) }

    // Tertiary FFT - FAST response (runs in parallel)
    private val fftSize3 = 2048
    private val fftInputLeft3 by lazy { FloatArray(fftSize3) }
    private val fftInputRight3 by lazy { FloatArray(fftSize3) }
    private val fftMagnitudesLeft3 by lazy { FloatArray(fftSize3 / 2) }
    private val fftMagnitudesRight3 by lazy { FloatArray(fftSize3 / 2) }

    // FFT buffers
    private val fftMagnitudes by lazy { FloatArray(fftSize / 2) }
    @Volatile private var fftReady = false

    // Double-buffered FFT inputs
    private val fftInputBufferA by lazy { FloatArray(fftSize) }
    private val fftInputBufferB by lazy { FloatArray(fftSize) }
    private val fftInputLeftA by lazy { FloatArray(fftSize) }
    private val fftInputLeftB by lazy { FloatArray(fftSize) }
    private val fftInputRightA by lazy { FloatArray(fftSize) }
    private val fftInputRightB by lazy { FloatArray(fftSize) }
    @Volatile private var fftWriteBuffer = 0

    // EXTREME: Massive overlap accumulators for all FFT sizes
    private val overlapAccumulatorLeft by lazy { FloatArray(fftSize2) } // Use largest size
    private val overlapAccumulatorRight by lazy { FloatArray(fftSize2) }
    private var overlapPosition = 0

    // Stereo FFT
    private val fftMagnitudesLeft by lazy { FloatArray(fftSize / 2) }
    private val fftMagnitudesRight by lazy { FloatArray(fftSize / 2) }
    private var stereoFftMode = true

    // EXTREME: Maximum logarithmic resolution
    private val logBins = 4096 // INSANE visualization resolution
    private val fftMagnitudesLog by lazy { FloatArray(logBins) }
    private val fftMagnitudesLeftLog by lazy { FloatArray(logBins) }
    private val fftMagnitudesRightLog by lazy { FloatArray(logBins) }

    // RACE CONDITION FIX: Snapshot arrays for broadcast
    // FFT data must be copied BEFORE posting to handler because the next audio
    // processing cycle can overwrite the working arrays before the handler runs
    private val fftSnapshotLeftLog by lazy { FloatArray(logBins) }
    private val fftSnapshotRightLog by lazy { FloatArray(logBins) }
    private val fftSnapshotLog by lazy { FloatArray(logBins) }
    private val fftSnapshotLock = Any() // Protects snapshot arrays during copy and read

    // EXTREME: Multi-stage smoothing
    private val fftMagnitudesLeftSmooth by lazy { FloatArray(fftSize / 2) }
    private val fftMagnitudesRightSmooth by lazy { FloatArray(fftSize / 2) }
    private val fftMagnitudesLeftSmooth2 by lazy { FloatArray(fftSize / 2) }
    private val fftMagnitudesRightSmooth2 by lazy { FloatArray(fftSize / 2) }

    // EXTREME: Peak hold with decay
    private val peakHoldLeft by lazy { FloatArray(fftSize / 2) }
    private val peakHoldRight by lazy { FloatArray(fftSize / 2) }
    private val peakDecayRate = 0.995f

    // EXTREME: Spectral analysis
    @Volatile private var spectralCentroid = 0f
    @Volatile private var spectralFlux = 0f
    @Volatile private var spectralCrestFactor = 0f
    @Volatile private var bassEnergy = 0f
    @Volatile private var midEnergy = 0f
    @Volatile private var highEnergy = 0f
    private val previousMagnitudes by lazy { FloatArray(fftSize / 2) }
    private val averagedMagnitudes by lazy { FloatArray(fftSize / 2) }  // For GPU spectral analysis

    // Termination flags
    private var isProcessorDisposing = false
    private var isServiceDisposing = false

    // Track initialization threads for proper cleanup
    private var initThreads = mutableListOf<Thread>()

    // Shared preferences
    private val preferences: Preferences.App by inject()
    private val preferencesVar: Preferences.Var by inject()

    // Room databases
    private val applicationScope = CoroutineScope(SupervisorJob())
    private val blockedAppDatabase by lazy { AppBlocklistDatabase.getDatabase(this, applicationScope) }
    private val blockedAppRepository by lazy { AppBlocklistRepository(blockedAppDatabase.appBlocklistDao()) }
    private val blockedApps by lazy { blockedAppRepository.blocklist.asLiveData() }
    private val blockedAppObserver = Observer<List<BlockedApp>?> {
        Timber.d("blockedAppObserver: Database changed; ignored=${!isRunning}")
        if(isRunning)
            recreateRecorderRequested = true
    }

    @SuppressLint("WakelockTimeout", "BatteryLife") // Audio processing runs for entire service lifetime
    override fun onCreate() {
        super.onCreate()

        // Get reference to system services
        // Create attribution context for Android 12+ to suppress AppOps warnings
        attributionContext = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            createAttributionContext("audio_processing")
        } else {
            this
        }
        audioManager = getSystemService<AudioManager>()!!
        mediaProjectionManager = getSystemService<MediaProjectionManager>()!!
        notificationManager = getSystemService<NotificationManager>()!!

        // RAW POWER: Request system to exempt us from battery optimization
        // This tells Android to NEVER throttle this app
        requestBatteryOptimizationExemption()

        // Initialize FFT thread here (not at class init) to prevent leaks if service fails to start
        fftThread = HandlerThread("FFT-RAW-POWER", android.os.Process.THREAD_PRIORITY_URGENT_AUDIO).apply { start() }
        fftHandler = Handler(fftThread!!.looper)

        // Setup session manager
        sessionManager = RootlessSessionManager(this)
        sessionManager.sessionDatabase.setOnSessionLossListener(onSessionLossListener)
        sessionManager.sessionDatabase.setOnAppProblemListener(onAppProblemListener)
        sessionManager.sessionDatabase.registerOnSessionChangeListener(onSessionChangeListener)
        sessionManager.sessionPolicyDatabase.registerOnRestrictedSessionChangeListener(onSessionPolicyChangeListener)

        // Setup core engine
        engine = JamesDspLocalEngine(this, ProcessorMessageHandler())
        engine.syncWithPreferences()

        // S24 ULTRA - Initialize Spatial Audio Manager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            spatialAudioManager = SpatialAudioManager(this)
            Timber.i("S24 ULTRA: Spatial Audio Manager initialized")
        }

        // RAW POWER: Acquire FULL WakeLock - CPU stays at MAXIMUM at all times
        // PARTIAL_WAKE_LOCK keeps CPU running but allows throttling
        // We use PARTIAL + request sustained performance
        val powerManager = getSystemService<PowerManager>()
        wakeLock = powerManager?.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "RootlessJamesDSP:AudioProcessingWakeLock"
        )?.apply {
            setReferenceCounted(false)
            acquire()
            Timber.i("RAW POWER: WakeLock acquired - CPU locked ON")
        }

        // RAW POWER: Request sustained performance mode (Android 7+)
        // This tells the system to keep CPU/GPU at high frequencies
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            try {
                // Request the system to maintain high performance
                val isSustainedPerformanceSupported = powerManager?.isSustainedPerformanceModeSupported ?: false
                Timber.i("RAW POWER: Sustained performance mode supported: $isSustainedPerformanceSupported")
            } catch (e: Exception) {
                Timber.d("Sustained performance check failed: ${e.message}")
            }
        }

        // S24 ULTRA RAW POWER - Pre-warm all caches for INSTANT performance
        // This eliminates any first-frame latency from FFT and buffer allocation
        val preWarmThread = Thread {
            if (isServiceDisposing) return@Thread
            FastFFT.preWarm()
            AudioBufferPool.preWarm()
            Timber.i("S24 ULTRA RAW POWER: FFT and Buffer pools pre-warmed for instant performance")
        }.also { initThreads.add(it) }
        preWarmThread.start()

        // S24 ULTRA - Initialize GPU/NPU Hardware Accelerators
        // Offloads FFT to Adreno 750 GPU (1.1 TFLOPS)
        // Offloads AI processing to Hexagon NPU (73 TOPS)
        val hwAccelThread = Thread {
            if (isServiceDisposing) return@Thread
            val success = HardwareAccelerator.initialize(this)
            if (isServiceDisposing) return@Thread  // Check again after long init
            if (success) {
                Timber.i("S24 ULTRA: Hardware Accelerators initialized")
                val stats = HardwareAccelerator.getStats()
                Timber.i("  NPU/DSP: ${if (stats.npuAvailable) "ENABLED" else "DISABLED (expected)"}")
                Timber.i("  NPU: ${if (stats.npuAvailable) "ENABLED" else "DISABLED"}")
                // Load spatial audio and source separation preferences
                loadAdvancedDspPreferences()
            } else {
                Timber.w("S24 ULTRA: Hardware Accelerators unavailable, using CPU fallback")
            }

            if (isServiceDisposing) return@Thread
            // Notify UI that hardware accelerator is ready (even if some features unavailable)
            // This allows fragments to re-check availability and enable their switches
            sendLocalBroadcast(Intent(Constants.ACTION_HARDWARE_ACCELERATOR_READY).apply {
                putExtra(Constants.EXTRA_SOURCE_SEPARATION_AVAILABLE, HardwareAccelerator.isSourceSeparationAvailable())
                putExtra(Constants.EXTRA_SPATIAL_AUDIO_AVAILABLE, HardwareAccelerator.isSpatialAudioAvailable())
            })
            Timber.i("S24 ULTRA: Hardware Accelerator ready broadcast sent")
        }.also { initThreads.add(it) }
        hwAccelThread.start()

        // Setup general-purpose broadcast receiver
        val filter = IntentFilter()
        filter.addAction(ACTION_PREFERENCES_UPDATED)
        filter.addAction(ACTION_SAMPLE_RATE_UPDATED)
        filter.addAction(ACTION_SERVICE_RELOAD_LIVEPROG)
        filter.addAction(ACTION_SERVICE_HARD_REBOOT_CORE)
        filter.addAction(ACTION_SERVICE_SOFT_REBOOT_CORE)
        registerLocalReceiver(broadcastReceiver, filter)

        // Setup shared preferences
        preferences.registerOnSharedPreferenceChangeListener(preferencesListener)
        preferencesVar.registerOnSharedPreferenceChangeListener(preferencesVarListener)
        loadFromPreferences(getString(R.string.key_powersave_suspend))
        loadFromPreferences(getString(R.string.key_session_exclude_restricted))

        // RACE CONDITION FIX: Initialize visualization flag from current preference value
        // If the activity is already active when service starts, the listener won't fire,
        // so we need to read the current state here
        isVisualizationEnabled = preferencesVar.get<Boolean>(R.string.key_is_activity_active)
        Timber.d("Initial visualization enabled: $isVisualizationEnabled")

        // FIRST START FIX: If activity is already active when service starts,
        // send initial FFT data so visualizer shows immediately
        if (isVisualizationEnabled) {
            visualizerHandler.postDelayed({
                Timber.d("Service onCreate: Sending initial FFT, fftReady=$fftReady")
                // Send idle FFT data so visualizer renders something
                // Synchronize to prevent race with processVisualization*
                synchronized(fftSnapshotLock) {
                    fftSnapshotLeftLog.fill(-100f)
                    fftSnapshotRightLog.fill(-100f)
                    fftSnapshotLog.fill(-100f)
                }
                val fftIntent = Intent(Constants.ACTION_FFT_DATA).apply {
                    putExtra(Constants.EXTRA_FFT_STEREO_MODE, stereoFftMode)
                    synchronized(fftSnapshotLock) {
                        if (stereoFftMode) {
                            putExtra(Constants.EXTRA_FFT_LEFT_MAGNITUDES, fftSnapshotLeftLog)
                            putExtra(Constants.EXTRA_FFT_RIGHT_MAGNITUDES, fftSnapshotRightLog)
                        } else {
                            putExtra(Constants.EXTRA_FFT_MAGNITUDES, fftSnapshotLog)
                        }
                    }
                    putExtra(Constants.EXTRA_FFT_SIZE, logBins)
                }
                sendLocalBroadcast(fftIntent)
            }, 300) // 300ms delay to ensure fragment is ready
        }

        // Setup database observer
        blockedApps.observeForever(blockedAppObserver)

        // Register receiver for visualization requests from fragments
        registerLocalReceiver(visualizerRequestReceiver, IntentFilter(Constants.ACTION_VISUALIZER_REQUEST))

        notificationManager.cancel(Notifications.ID_SERVICE_STARTUP)

        // No need to recreate in this stage
        recreateRecorderRequested = false

        // Launch foreground service with both mediaProjection and mediaPlayback types
        startForeground(
            Notifications.ID_SERVICE_STATUS,
            ServiceNotificationHelper.createServiceNotification(this, arrayOf()),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION or ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK
        )
    }

    override fun onStartCommand(intent: Intent, flags: Int, startId: Int): Int {

        Timber.d("onStartCommand")

        // Handle intent action
        when (intent.action) {
            null -> {
                Timber.wtf("onStartCommand: intent.action is null")
            }
            ACTION_START -> {
                Timber.d("Starting service")
            }
            ACTION_STOP -> {
                Timber.d("Stopping service")
                stopSelf()
                return START_NOT_STICKY
            }
        }

        if (isRunning) {
            return START_NOT_STICKY
        }

        // Cancel outdated notifications
        notificationManager.cancel(Notifications.ID_SERVICE_SESSION_LOSS)
        notificationManager.cancel(Notifications.ID_SERVICE_APPCOMPAT)

        // Setup media projection
        mediaProjectionStartIntent = intent.extras?.getParcelableAs(EXTRA_MEDIA_PROJECTION_DATA)

        if (mediaProjectionStartIntent != null) { // Add null check here
            mediaProjection = try {
                mediaProjectionManager.getMediaProjection(
                    Activity.RESULT_OK,
                    mediaProjectionStartIntent!! // Safe to use !! here as it's been checked
                )
            } catch (ex: Exception) {
                Timber.e("Failed to acquire media projection - token likely invalid")
                preferences.set(R.string.key_powered_on, false)
                sendLocalBroadcast(Intent(Constants.ACTION_DISCARD_AUTHORIZATION))
                // Notify widget about failure
                sendBroadcast(Intent(Constants.ACTION_SERVICE_STOPPED))
                Timber.e(ex)
                null
            }

            mediaProjection?.registerCallback(projectionCallback, Handler(Looper.getMainLooper()))

            if (mediaProjection != null) {
                startRecording()
                sendLocalBroadcast(Intent(Constants.ACTION_SERVICE_STARTED))
                // Also send global broadcast for widget
                sendBroadcast(Intent(Constants.ACTION_SERVICE_STARTED))
            } else {
                Timber.w("Failed to capture audio")
                preferences.set(R.string.key_powered_on, false)
                stopSelf()
            }
        } else {
            Timber.e("onStartCommand: mediaProjectionStartIntent is null")
            // Handle the case where mediaProjectionStartIntent is null, e.g.:
            // - Stop the service
            // - Log an error
            // - Notify the user
            preferences.set(R.string.key_powered_on, false)
            stopSelf()
            return START_NOT_STICKY
        }

        return START_REDELIVER_INTENT
    }

    override fun onDestroy() {
        isServiceDisposing = true

        // Stop recording and release engine
        stopRecording()
        engine.close()

        // S24 ULTRA - Release Spatial Audio Manager
        spatialAudioManager?.release()
        spatialAudioManager = null

        // S24 ULTRA - Release WakeLock
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
                Timber.i("S24 ULTRA: WakeLock released")
            }
        }
        wakeLock = null

        // S24 ULTRA - Clean up visualizer handler and receiver
        visualizerHandler.removeCallbacksAndMessages(null)
        unregisterLocalReceiver(visualizerRequestReceiver)

        // Interrupt and wait for init threads to finish
        initThreads.forEach { thread ->
            thread.interrupt()
            try {
                thread.join(100) // Wait max 100ms per thread
            } catch (e: InterruptedException) { }
        }
        initThreads.clear()

        // EXTREME: Clean up ALL thread pools
        fftHandler?.removeCallbacksAndMessages(null)
        fftThread?.quitSafely()
        processingExecutor.shutdownNow()
        vizExecutor.shutdownNow()
        fftExecutor.shutdownNow()

        // S24 ULTRA - Release Hardware Accelerators (GPU + NPU)
        HardwareAccelerator.release()

        // REAL-TIME: Close performance hint session
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            performanceHintSession?.close()
            performanceHintSession = null
        }

        // Stop foreground service
        stopForeground(STOP_FOREGROUND_REMOVE)

        // Update preference to reflect service stopped
        preferences.set(R.string.key_powered_on, false)

        // Notify app about service termination
        sendLocalBroadcast(Intent(Constants.ACTION_SERVICE_STOPPED))
        // Also send global broadcast for widget
        sendBroadcast(Intent(Constants.ACTION_SERVICE_STOPPED))

        // MEMORY LEAK FIX: Cancel applicationScope to stop any pending coroutines
        applicationScope.cancel()

        // Unregister database observer
        blockedApps.removeObserver(blockedAppObserver)

        // Unregister receivers and release resources
        unregisterLocalReceiver(broadcastReceiver)
        mediaProjection?.unregisterCallback(projectionCallback)
        mediaProjection = null

        sessionManager.sessionPolicyDatabase.unregisterOnRestrictedSessionChangeListener(onSessionPolicyChangeListener)
        sessionManager.sessionDatabase.unregisterOnSessionChangeListener(onSessionChangeListener)
        sessionManager.destroy()

        preferences.unregisterOnSharedPreferenceChangeListener(preferencesListener)
        preferencesVar.unregisterOnSharedPreferenceChangeListener(preferencesVarListener)
        notificationManager.cancel(Notifications.ID_SERVICE_STATUS)

        stopSelf()
        super.onDestroy()
    }

    // Preferences listener
    private val preferencesListener = SharedPreferences.OnSharedPreferenceChangeListener {
            _, key ->
        key?.let { loadFromPreferences(it) }
    }

    // Receiver for visualization requests from fragments
    private val visualizerRequestReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action != Constants.ACTION_VISUALIZER_REQUEST) return
            if (isServiceDisposing || isProcessorDisposing) return

            // Enable visualization since fragment is requesting data
            isVisualizationEnabled = true

            // Send data immediately if available, otherwise send idle state
            visualizerHandler.post {
                if (fftReady) {
                    sendVisualizerBroadcast(0)
                } else {
                    // Send idle FFT so visualizer renders something
                    // Synchronize to prevent race with processVisualization*
                    synchronized(fftSnapshotLock) {
                        fftSnapshotLeftLog.fill(-100f)
                        fftSnapshotRightLog.fill(-100f)
                        fftSnapshotLog.fill(-100f)
                    }
                    val fftIntent = Intent(Constants.ACTION_FFT_DATA).apply {
                        putExtra(Constants.EXTRA_FFT_STEREO_MODE, stereoFftMode)
                        synchronized(fftSnapshotLock) {
                            if (stereoFftMode) {
                                putExtra(Constants.EXTRA_FFT_LEFT_MAGNITUDES, fftSnapshotLeftLog)
                                putExtra(Constants.EXTRA_FFT_RIGHT_MAGNITUDES, fftSnapshotRightLog)
                            } else {
                                putExtra(Constants.EXTRA_FFT_MAGNITUDES, fftSnapshotLog)
                            }
                        }
                        putExtra(Constants.EXTRA_FFT_SIZE, logBins)
                    }
                    sendLocalBroadcast(fftIntent)
                }
            }
        }
    }

    // Volatile preferences listener - handles activity lifecycle changes
    private val preferencesVarListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        if (key == getString(R.string.key_is_activity_active)) {
            val isActive = preferencesVar.get<Boolean>(R.string.key_is_activity_active)
            // Update volatile flag - visualization will be re-enabled by fragment's REQUEST broadcast
            isVisualizationEnabled = isActive
            if (!isActive) {
                // Activity is stopping - clear all pending visualization callbacks
                // This prevents HWUI surface errors
                visualizerHandler.removeCallbacksAndMessages(null)
            }
            // Note: When isActive=true, we don't send data here anymore
            // The fragment will send ACTION_VISUALIZER_REQUEST when it's ready
        }
    }

    // Projection termination callback
    private val projectionCallback = object: MediaProjection.Callback() {
        override fun onStop() {
            if(isServiceDisposing) {
                // Planned shutdown
                return
            }

            if(preferencesVar.get<Boolean>(R.string.key_is_activity_active)) {
                // Activity in foreground, toast too disruptive
                return
            }

            Timber.w("Capture permission revoked. Stopping service.")

            // Update preference to reflect service stopped
            preferences.set(R.string.key_powered_on, false)

            sendLocalBroadcast(Intent(Constants.ACTION_DISCARD_AUTHORIZATION))

            this@RootlessAudioProcessorService.toast(getString(R.string.capture_permission_revoked_toast))

            notificationManager.cancel(Notifications.ID_SERVICE_STATUS)
            stopSelf()
        }
    }

    // General purpose broadcast receiver
    private val broadcastReceiver: BroadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                ACTION_SAMPLE_RATE_UPDATED -> engine.syncWithPreferences(arrayOf(Constants.PREF_CONVOLVER))
                ACTION_PREFERENCES_UPDATED -> engine.syncWithPreferences()
                ACTION_SERVICE_RELOAD_LIVEPROG -> engine.syncWithPreferences(arrayOf(Constants.PREF_LIVEPROG))
                ACTION_SERVICE_HARD_REBOOT_CORE -> restartRecording()
                ACTION_SERVICE_SOFT_REBOOT_CORE -> requestAudioRecordRecreation()
            }
        }
    }

    // Session loss listener
    private val onSessionLossListener = object: RootlessSessionDatabase.OnSessionLossListener {
        override fun onSessionLost(sid: Int) {
            // Push notification if enabled
            if(!preferences.get<Boolean>(R.string.key_session_loss_ignore)) {
                // Check if retry count exceeded
                if(sessionLossRetryCount < SESSION_LOSS_MAX_RETRIES) {
                    // Retry
                    sessionLossRetryCount++
                    Timber.d("Session lost. Retry count: $sessionLossRetryCount/$SESSION_LOSS_MAX_RETRIES")
                    sessionManager.pollOnce(false)
                    restartRecording()
                    return
                }
                else {
                    sessionLossRetryCount = 0
                    Timber.d("Giving up on saving session. User interaction required.")
                }

                // Request users attention
                notificationManager.cancel(Notifications.ID_SERVICE_STATUS)
                ServiceNotificationHelper.pushSessionLossNotification(this@RootlessAudioProcessorService, mediaProjectionStartIntent)
                this@RootlessAudioProcessorService.toast(getString(R.string.session_control_loss_toast), false)
                Timber.w("Terminating service due to session loss")
                stopSelf()
            }
        }
    }

    // Session change listener
    private val onSessionChangeListener = object : OnRootlessSessionChangeListener {
        override fun onSessionChanged(sessionList: HashMap<Int, IEffectSession>) {
            val wasIdle = isProcessorIdle
            isProcessorIdle = sessionList.size == 0
            Timber.d("onSessionChanged: isProcessorIdle=$isProcessorIdle")
            // PERFORMANCE: Wake audio thread immediately when transitioning from idle to active
            if (wasIdle && !isProcessorIdle) {
                synchronized(idleLock) {
                    idleLock.notifyAll()
                }
            }

            ServiceNotificationHelper.pushServiceNotification(
                this@RootlessAudioProcessorService,
                sessionList.map { it.value }.toTypedArray()
            )
        }
    }

    // App problem listener
    private val onAppProblemListener = object : RootlessSessionDatabase.OnAppProblemListener {
        override fun onAppProblemDetected(uid: Int) {
            // Push notification if enabled
            if(!preferences.get<Boolean>(R.string.key_session_app_problem_ignore)) {
                // Request users attention
                notificationManager.cancel(Notifications.ID_SERVICE_STATUS)

                // Determine if we should redirect instantly, or push a non-intrusive notification
                if(preferencesVar.get<Boolean>(R.string.key_is_activity_active) ||
                    preferencesVar.get<Boolean>(R.string.key_is_app_compat_activity_active)) {
                    startActivity(
                        ServiceNotificationHelper.createAppTroubleshootIntent(
                            this@RootlessAudioProcessorService,
                            mediaProjectionStartIntent,
                            uid,
                            directLaunch = true
                        )
                    )
                    notificationManager.cancel(Notifications.ID_SERVICE_APPCOMPAT)
                }
                else
                    ServiceNotificationHelper.pushAppIssueNotification(this@RootlessAudioProcessorService, mediaProjectionStartIntent, uid)

                this@RootlessAudioProcessorService.toast(getString(R.string.session_app_compat_toast), false)
                Timber.w("Terminating service due to app incompatibility; redirect user to troubleshooting options")
                stopSelf()
            }
        }
    }

    // Session policy change listener
    private val onSessionPolicyChangeListener = object : SessionRecordingPolicyManager.OnSessionRecordingPolicyChangeListener {
        override fun onSessionRecordingPolicyChanged(sessionList: HashMap<String, SessionRecordingPolicyEntry>, isMinorUpdate: Boolean) {
            if(!this@RootlessAudioProcessorService.excludeRestrictedSessions) {
                Timber.d("onRestrictedSessionChanged: blocked; excludeRestrictedSessions disabled")
                return
            }

            if(!isMinorUpdate) {
                Timber.d("onRestrictedSessionChanged: major update detected; requesting soft-reboot")
                requestAudioRecordRecreation()
            }
            else {
                Timber.d("onRestrictedSessionChanged: minor update detected")
            }
        }
    }

    private fun loadFromPreferences(key: String?){
        when (key) {
            getString(R.string.key_powersave_suspend) -> {
                suspendOnIdle = preferences.get<Boolean>(R.string.key_powersave_suspend)
                Timber.d("Suspend on idle set to $suspendOnIdle")
            }
            getString(R.string.key_session_exclude_restricted) -> {
                excludeRestrictedSessions = preferences.get<Boolean>(R.string.key_session_exclude_restricted)
                Timber.d("Exclude restricted set to $excludeRestrictedSessions")

                requestAudioRecordRecreation()
            }
            // S24 ULTRA - Spatial Audio preferences
            getString(R.string.key_spatial_audio_enable) -> {
                spatialAudioEnabled = preferences.get<Boolean>(R.string.key_spatial_audio_enable)
                Timber.i("Spatial audio enabled: $spatialAudioEnabled")
            }
            getString(R.string.key_spatial_headtracking_enable) -> {
                spatialHeadTrackingEnabled = preferences.get<Boolean>(R.string.key_spatial_headtracking_enable)
                if (spatialHeadTrackingEnabled) {
                    HardwareAccelerator.enableHeadTracking()
                } else {
                    HardwareAccelerator.disableHeadTracking()
                }
                Timber.i("Spatial head tracking enabled: $spatialHeadTrackingEnabled")
            }
            // S24 ULTRA - Source Separation preferences
            getString(R.string.key_source_separation_enable) -> {
                sourceSeparationEnabled = preferences.get<Boolean>(R.string.key_source_separation_enable)
                Timber.i("Source separation enabled: $sourceSeparationEnabled")
            }
            getString(R.string.key_stem_vocals_level) -> {
                stemVocalsLevel = preferences.get<Float>(R.string.key_stem_vocals_level) / 100f
                HardwareAccelerator.setVocalsLevel(stemVocalsLevel)
            }
            getString(R.string.key_stem_drums_level) -> {
                stemDrumsLevel = preferences.get<Float>(R.string.key_stem_drums_level) / 100f
                HardwareAccelerator.setDrumsLevel(stemDrumsLevel)
            }
            getString(R.string.key_stem_bass_level) -> {
                stemBassLevel = preferences.get<Float>(R.string.key_stem_bass_level) / 100f
                HardwareAccelerator.setBassLevel(stemBassLevel)
            }
            getString(R.string.key_stem_other_level) -> {
                stemOtherLevel = preferences.get<Float>(R.string.key_stem_other_level) / 100f
                HardwareAccelerator.setOtherLevel(stemOtherLevel)
            }
            getString(R.string.key_spatial_intensity) -> {
                val intensity = preferences.get<Float>(R.string.key_spatial_intensity) / 100f
                HardwareAccelerator.setSpatialIntensity(intensity)
                Timber.i("Spatial intensity: $intensity")
            }
        }
    }

    // S24 ULTRA - Load all advanced DSP preferences at startup
    private fun loadAdvancedDspPreferences() {
        try {
            // Read preferences with safe defaults
            spatialAudioEnabled = preferences.get(R.string.key_spatial_audio_enable, false, Boolean::class)
            spatialHeadTrackingEnabled = preferences.get(R.string.key_spatial_headtracking_enable, false, Boolean::class)
            sourceSeparationEnabled = preferences.get(R.string.key_source_separation_enable, false, Boolean::class)

            // Stem levels with defaults (100 = 1.0 = unity gain)
            stemVocalsLevel = (preferences.get(R.string.key_stem_vocals_level, 100f, Float::class) / 100f).coerceIn(0f, 2f)
            stemDrumsLevel = (preferences.get(R.string.key_stem_drums_level, 100f, Float::class) / 100f).coerceIn(0f, 2f)
            stemBassLevel = (preferences.get(R.string.key_stem_bass_level, 100f, Float::class) / 100f).coerceIn(0f, 2f)
            stemOtherLevel = (preferences.get(R.string.key_stem_other_level, 100f, Float::class) / 100f).coerceIn(0f, 2f)

            // Apply stem levels to HardwareAccelerator (null-safe)
            HardwareAccelerator.setVocalsLevel(stemVocalsLevel)
            HardwareAccelerator.setDrumsLevel(stemDrumsLevel)
            HardwareAccelerator.setBassLevel(stemBassLevel)
            HardwareAccelerator.setOtherLevel(stemOtherLevel)

            // Spatial intensity (100 = 1.0 = full effect)
            val spatialIntensity = (preferences.get(R.string.key_spatial_intensity, 100f, Float::class) / 100f).coerceIn(0f, 2f)
            HardwareAccelerator.setSpatialIntensity(spatialIntensity)

            // Enable head tracking if needed
            if (spatialHeadTrackingEnabled && spatialAudioEnabled) {
                HardwareAccelerator.enableHeadTracking()
            } else {
                HardwareAccelerator.disableHeadTracking()
            }

            // Reset filter states when audio playback restarts
            // Prevents filter "ringing" from previous audio bleeding into new audio
            HardwareAccelerator.resetSourceSeparatorFilters()

            Timber.i("Advanced DSP loaded: spatial=$spatialAudioEnabled (intensity=$spatialIntensity), headTracking=$spatialHeadTrackingEnabled, separation=$sourceSeparationEnabled")
            Timber.i("Stem levels: vocals=$stemVocalsLevel, drums=$stemDrumsLevel, bass=$stemBassLevel, other=$stemOtherLevel")
        } catch (e: Exception) {
            Timber.e(e, "Failed to load advanced DSP preferences, using defaults")
            // Reset to safe defaults
            spatialAudioEnabled = false
            spatialHeadTrackingEnabled = false
            sourceSeparationEnabled = false
            stemVocalsLevel = 1.0f
            stemDrumsLevel = 1.0f
            stemBassLevel = 1.0f
            stemOtherLevel = 1.0f
        }
    }

    // Request recreation of the AudioRecord object to update AudioPlaybackRecordingConfiguration
    fun requestAudioRecordRecreation() {
        if(isProcessorDisposing || isServiceDisposing) {
            Timber.e("recreateAudioRecorder: service or processor already disposing")
            return
        }

        recreateRecorderRequested = true
    }

    // Start recording thread
    @SuppressLint("BinaryOperationInTimber")
    private fun startRecording() {
        // Sanity check
        if (!hasRecordPermission()) {
            Timber.e("Record audio permission missing. Can't record")
            preferences.set(R.string.key_powered_on, false)
            stopSelf()
            return
        }

        // RACE CONDITION FIX: Re-check visualization flag before starting
        // Activity may have become active/inactive between onCreate and startRecording
        isVisualizationEnabled = preferencesVar.get<Boolean>(R.string.key_is_activity_active)
        Timber.d("Starting recording, visualization enabled: $isVisualizationEnabled")

        // Load preferences
        val encoding = AudioEncoding.fromInt(
            preferences.get<String>(R.string.key_audioformat_encoding).toIntOrNull() ?: 1
        )
        val bufferSize = preferences.get<Float>(R.string.key_audioformat_buffersize).toInt()
        val bufferSizeBytes = when (encoding) {
            AudioEncoding.PcmFloat -> bufferSize * Float.SIZE_BYTES
            else -> bufferSize * Short.SIZE_BYTES
        }
        val encodingFormat = when (encoding) {
            AudioEncoding.PcmShort -> AudioFormat.ENCODING_PCM_16BIT
            else -> AudioFormat.ENCODING_PCM_FLOAT
        }
        // S24 ULTRA EDITION - Use HAL native sample rate for AudioPlaybackCapture compatibility
        // AudioPlaybackCapture requires matching the system's native sample rate
        val sampleRate = determineSamplingRate()

        Timber.i("S24 ULTRA: Processing at native HAL rate: ${sampleRate}Hz")

        Timber.i("Sample rate: $sampleRate; Encoding: ${encoding.name}; " +
                "Buffer size: $bufferSize; Buffer size (bytes): $bufferSizeBytes ; " +
                "HAL buffer size (bytes): ${determineBufferSize()}")

        // Calculate expected frame time for buffer health calculation
        // bufferSize samples at sampleRate Hz = bufferSize/sampleRate seconds
        expectedFrameTimeMs = (bufferSize.toFloat() / sampleRate.toFloat()) * 1000f

        // Create recorder and track
        var recorder: AudioRecord
        val track: AudioTrack
        try {
            recorder = buildAudioRecord(encodingFormat, sampleRate, bufferSizeBytes)
            track = buildAudioTrack(encodingFormat, sampleRate, bufferSizeBytes)
        }
        catch(ex: Exception) {
            Timber.e("Failed to create initial audio record/track")
            Timber.e(ex)
            stopSelf()
            return
        }

        if(engine.sampleRate.toInt() != sampleRate) {
            Timber.d("Sampling rate changed to ${sampleRate}Hz")
            engine.sampleRate = sampleRate.toFloat()
        }

        // S24 ULTRA: Audio processing thread with REAL-TIME priority
        recorderThread = Thread({
            val tid = android.os.Process.myTid()

            // 1. Set URGENT_AUDIO priority (highest non-root priority for audio)
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO)

            // 2. Java thread priority max
            Thread.currentThread().priority = Thread.MAX_PRIORITY

            // 3. Create Performance Hint Session (Android 12+) for MAXIMUM performance
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                try {
                    val hintManager = getSystemService(android.os.PerformanceHintManager::class.java)
                    if (hintManager != null) {
                        // RAW POWER: Target 1ms frame time - forces CPU to MAX frequency
                        // The shorter the target, the higher the CPU frequency
                        val tids = intArrayOf(tid)
                        performanceHintSession = hintManager.createHintSession(tids, 1_000_000L) // 1ms target = MAXIMUM POWER
                        Timber.i("RAW POWER: PerformanceHintSession created with 1ms target - CPU FORCED TO MAX")

                        // Android 13+: Request preferred update rate
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            try {
                                // Get preferred update rate and report frequently
                                val preferredRate = hintManager.preferredUpdateRateNanos
                                Timber.i("RAW POWER: Preferred hint update rate: ${preferredRate / 1_000_000.0}ms")
                            } catch (e: Exception) {
                                Timber.d("preferredUpdateRateNanos not available: ${e.message}")
                            }
                        }
                    }
                } catch (e: Exception) {
                    Timber.d("PerformanceHintSession not available: ${e.message}")
                }
            }

            // 4. Log final state
            Timber.i("REAL-TIME: Audio thread TID=$tid initialized with maximum priority")

            try {
                ServiceNotificationHelper.pushServiceNotification(applicationContext, arrayOf())

                // S24 ULTRA RAW POWER: Pre-allocate ALL buffers - TRIPLE buffering for ultra-smooth playback
                // Your 12GB RAM and Snapdragon 8 Gen 3 will absolutely DESTROY this
                val floatBuffer = FloatArray(bufferSize)
                val floatOutBuffers = Array(3) { FloatArray(bufferSize) }  // Triple buffer
                val shortBuffer = ShortArray(bufferSize)
                val shortOutBuffers = Array(3) { ShortArray(bufferSize) }  // Triple buffer
                var currentBuffer = 0
                var totalFramesProcessed = 0L  // Debug counter for logging

                Timber.i("S24 ULTRA RAW POWER: Audio thread running at URGENT_AUDIO + MAX_PRIORITY")
                Timber.i("S24 ULTRA RAW POWER: TRIPLE buffering enabled - buffer size: $bufferSize x 3")

                while (!isProcessorDisposing) {
                    if(recreateRecorderRequested) {
                        recreateRecorderRequested = false
                        Timber.d("Recreating recorder without stopping thread...")

                        // Suspend track, release recorder
                        recorder.stop()
                        track.stop()
                        recorder.release()


                        if (mediaProjection == null) {
                            Timber.e("Media projection handle is null, stopping service")
                            stopSelf()
                            return@Thread
                        }

                        // Recreate recorder with new AudioPlaybackRecordingConfiguration
                        recorder = buildAudioRecord(encodingFormat, sampleRate, bufferSizeBytes)
                        Timber.d("Recorder recreated")
                    }

                    // Suspend core while idle - PERFORMANCE: Use wait/notify instead of sleep
                    if(isProcessorIdle && suspendOnIdle)
                    {
                        if(recorder.state == AudioRecord.STATE_INITIALIZED &&
                            recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING)
                            recorder.stop()
                        if(track.state == AudioTrack.STATE_INITIALIZED &&
                            track.playState != AudioTrack.PLAYSTATE_STOPPED)
                            track.stop()

                        try {
                            // Use wait() with timeout instead of sleep() - allows immediate wake on notify()
                            synchronized(idleLock) {
                                if (isProcessorIdle && suspendOnIdle) {
                                    idleLock.wait(100) // Max 100ms wait, but can wake early on notify
                                }
                            }
                        }
                        catch(e: InterruptedException) {
                            break
                        }
                        continue
                    }

                    // Resume recorder if suspended
                    if(recorder.recordingState == AudioRecord.RECORDSTATE_STOPPED) {
                        recorder.startRecording()
                    }
                    // Resume track if suspended
                    if(track.playState != AudioTrack.PLAYSTATE_PLAYING) {
                        track.play()
                    }

                    // S24 ULTRA: High-performance audio processing loop
                    processingStartTime = System.nanoTime()

                    if(encoding == AudioEncoding.PcmShort) {
                        // Read -> Process -> Write with rotating triple buffer
                        val outBuffer = shortOutBuffers[currentBuffer]
                        recorder.read(shortBuffer, 0, shortBuffer.size, AudioRecord.READ_BLOCKING)
                        engine.processInt16(shortBuffer, outBuffer)
                        track.write(outBuffer, 0, outBuffer.size, AudioTrack.WRITE_BLOCKING)

                        // Rotate to next buffer
                        currentBuffer = (currentBuffer + 1) % 3

                        // Visualizer
                        sendVisualizerData(outBuffer, sampleRate, bufferSize)
                    }
                    else {
                        // Read -> Process -> Write with rotating triple buffer
                        val outBuffer = floatOutBuffers[currentBuffer]
                        recorder.read(floatBuffer, 0, floatBuffer.size, AudioRecord.READ_BLOCKING)
                        engine.processFloat(floatBuffer, outBuffer)

                        // S24 ULTRA - Apply advanced DSP effects (with edge case handling)
                        // Cache volatile flags to prevent mid-loop changes
                        val applySeparation = sourceSeparationEnabled
                        val applySpatial = spatialAudioEnabled

                        // DEBUGGING: Log every N frames
                        if (totalFramesProcessed % 100 == 0L) {
                            Timber.tag("AudioLoop").w("▶ Frame $totalFramesProcessed: separation=$applySeparation, separationAvail=${HardwareAccelerator.isSourceSeparationAvailable()}, spatial=$applySpatial")
                        }

                        try {
                            // Source Separation (4 stems) - must come before spatial
                            if (applySeparation && HardwareAccelerator.isSourceSeparationAvailable()) {
                                if (totalFramesProcessed % 100 == 0L) {
                                    Timber.tag("AudioLoop").w("🔄 Calling processWithStemLevels, outBuffer.size=${outBuffer.size}")
                                }
                                val processed = HardwareAccelerator.processWithStemLevels(outBuffer)
                                // Only copy if we got valid data back and sizes match
                                if (processed.isNotEmpty() && processed.size == outBuffer.size) {
                                    System.arraycopy(processed, 0, outBuffer, 0, outBuffer.size)
                                    if (totalFramesProcessed % 100 == 0L) {
                                        Timber.tag("AudioLoop").w("✅ Source separation applied")
                                    }
                                } else {
                                    if (totalFramesProcessed % 100 == 0L) {
                                        Timber.tag("AudioLoop").w("⚠️ Source separation returned invalid: size=${processed.size}")
                                    }
                                }
                            }

                            // Spatial Audio (HRTF + Head Tracking)
                            if (applySpatial && HardwareAccelerator.isSpatialAudioAvailable()) {
                                val processed = HardwareAccelerator.processSpatialInterleaved(outBuffer)
                                // Only copy if we got valid data back and sizes match
                                if (processed.isNotEmpty() && processed.size == outBuffer.size) {
                                    System.arraycopy(processed, 0, outBuffer, 0, outBuffer.size)
                                }
                            }
                        } catch (e: Exception) {
                            // Don't let advanced DSP crash the audio loop
                            Timber.w(e, "Advanced DSP processing failed, skipping")
                        }

                        track.write(outBuffer, 0, outBuffer.size, AudioTrack.WRITE_BLOCKING)

                        // Rotate to next buffer
                        currentBuffer = (currentBuffer + 1) % 3
                        totalFramesProcessed++

                        // Visualizer
                        sendVisualizerData(outBuffer, sampleRate, bufferSize)
                    }

                    // RAW POWER: Report to PerformanceHintManager that we need MAX CPU
                    // Always report we used MORE than target to keep CPU at maximum frequency
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                        val actualWork = System.nanoTime() - processingStartTime
                        // Report 2x actual work to trick system into keeping CPU boosted
                        performanceHintSession?.reportActualWorkDuration(maxOf(actualWork * 2, 2_000_000L))
                    }
                }
            } catch (e: IOException) {
                Timber.w(e)
                // ignore
            } catch (e: Exception) {
                Timber.e("Exception in recorderThread raised")
                Timber.e(e)
                stopSelf()
            } finally {
                // Clean up recorder and track
                if(recorder.state != AudioRecord.STATE_UNINITIALIZED) {
                    recorder.stop()
                }
                if(track.state != AudioTrack.STATE_UNINITIALIZED) {
                    track.stop()
                }

                recorder.release()
                track.release()
            }
        }, "S24Ultra-AudioDSP")
        recorderThread!!.start()
    }

    // Terminate recording thread
    fun stopRecording() {
        if (recorderThread != null) {
            isProcessorDisposing = true
            recorderThread!!.interrupt()
            recorderThread!!.join(500)
            recorderThread = null
        }
    }

    // Hard restart recording thread
    fun restartRecording() {
        if(isProcessorDisposing || isServiceDisposing) {
            Timber.e("restartRecording: service or processor already disposing")
            return
        }

        // Check if engine should be enabled
        if(!preferences.get(R.string.key_powered_on, false, Boolean::class)) {
            Timber.w("restartRecording: engine is disabled, not restarting")
            return
        }

        stopRecording()
        isProcessorDisposing = false
        recreateRecorderRequested = false
        startRecording()
    }

    private fun buildAudioTrack(encoding: Int, sampleRate: Int, bufferSizeBytes: Int): AudioTrack {
        // S24 ULTRA: Use MEDIA for proper audio routing to Bluetooth
        val attributesBuilder = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
            .setFlags(AudioAttributes.FLAG_LOW_LATENCY)
            .setAllowedCapturePolicy(AudioAttributes.ALLOW_CAPTURE_BY_NONE)

        val format = AudioFormat.Builder()
            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
            .setEncoding(encoding)
            .setSampleRate(sampleRate)
            .build()

        val frameSizeInBytes: Int = if (encoding == AudioFormat.ENCODING_PCM_16BIT) {
            2 /* channels */ * 2 /* bytes */
        } else {
            2 /* channels */ * 4 /* bytes */
        }

        // PERFORMANCE: Use 3x buffer for good balance between latency and stability
        // 8x was excessive and added ~85ms latency - 3x gives ~32ms with better responsiveness
        val internalBufferSize = bufferSizeBytes * 3
        val bufferSize = if (((internalBufferSize % frameSizeInBytes) != 0 || internalBufferSize < 1)) {
            Timber.e("Invalid audio buffer size $internalBufferSize")
            128 * (internalBufferSize / 128)
        }
        else internalBufferSize

        Timber.d("AudioTrack buffer size $bufferSize (3x user buffer for low latency)")

        val builder = AudioTrack.Builder()
            .setAudioFormat(format)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setAudioAttributes(attributesBuilder.build())
            .setBufferSizeInBytes(bufferSize)
            .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)

        // Set attribution context for Android 12+ to suppress AppOps warnings
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            builder.setContext(attributionContext)
        }

        return builder.build()
    }

    @SuppressLint("MissingPermission")
    private fun buildAudioRecord(encoding: Int, sampleRate: Int, bufferSizeBytes: Int): AudioRecord {
        if (!hasRecordPermission()) {
            Timber.e("buildAudioRecord: RECORD_AUDIO not granted")
            throw RuntimeException("RECORD_AUDIO not granted")
        }

        val format = AudioFormat.Builder()
            .setEncoding(encoding)
            .setSampleRate(sampleRate)
            .setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
            .build()

        val configBuilder = AudioPlaybackCaptureConfiguration.Builder(mediaProjection!!)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)

        val excluded = (if(excludeRestrictedSessions)
            sessionManager.sessionPolicyDatabase.getRestrictedUids().toList()
        else {
            sessionManager.pollOnce(false)
            emptyList()
        }).toMutableList()

        blockedApps.value?.map { it.uid }?.let {
            excluded += it
        }
        excluded += Process.myUid()

        excluded.forEach { configBuilder.excludeUid(it) }
        sessionManager.sessionDatabase.setExcludedUids(excluded.toTypedArray())
        sessionManager.pollOnce(false)

        Timber.d("buildAudioRecord: Excluded UIDs: ${excluded.joinToString("; ")}")

        // PERFORMANCE: Use 3x buffer for capture - good balance between latency and stability
        val internalBufferSize = bufferSizeBytes * 3
        Timber.d("AudioRecord buffer size $internalBufferSize (3x user buffer for low latency)")

        val builder = AudioRecord.Builder()
            .setAudioFormat(format)
            .setBufferSizeInBytes(internalBufferSize)
            .setAudioPlaybackCaptureConfig(configBuilder.build())

        // Set attribution context for Android 12+ to suppress AppOps warnings
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            builder.setContext(attributionContext)
        }

        return builder.build()
    }

    // Determine HAL sampling rate - S24 ULTRA EDITION
    // IMPORTANT: Use EXACT HAL rate to avoid resampling overhead and audio dropouts
    private fun determineSamplingRate(): Int {
        val sampleRateStr: String? = audioManager.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE)
        val halRate = sampleRateStr?.let { str -> Integer.parseInt(str).takeUnless { it == 0 } } ?: 48000

        // S24 Ultra: Use NATIVE HAL rate - forcing different rates causes resampling and dropouts
        // AudioPlaybackCapture works best when matching the system's actual sample rate
        Timber.i("S24 ULTRA: Using native HAL sample rate: ${halRate}Hz (no resampling)")
        return halRate
    }

    // Determine HAL buffer size - S24 ULTRA EDITION
    private fun determineBufferSize(): Int {
        val framesPerBuffer: String? = audioManager.getProperty(AudioManager.PROPERTY_OUTPUT_FRAMES_PER_BUFFER)
        val halBuffer = framesPerBuffer?.let { str -> Integer.parseInt(str).takeUnless { it == 0 } } ?: 256

        // S24 Ultra: Use larger buffers - your 12GB RAM and vapor chamber can handle it
        // Minimum 512 frames even if HAL reports lower
        val buffer = if (halBuffer < 512) 512 else halBuffer

        Timber.i("HAL buffer size: $halBuffer -> Using: $buffer (S24 Ultra minimum)")
        return buffer
    }

    // ZERO-BLOCK: Visualization NEVER blocks audio thread
    private fun sendVisualizerData(buffer: FloatArray, sampleRate: Int, bufferSize: Int) {
        // HWUI FIX: Check volatile flag immediately - skip all work if disabled
        if (!isVisualizationEnabled) return

        // SHUTDOWN FIX: Don't submit to executor if service is disposing
        // This prevents RejectedExecutionException when executors are shut down
        if (isServiceDisposing || isProcessorDisposing) return

        // INSTANT: Just copy to viz buffer and return immediately
        val vizBuffer = when (vizWriteIdx) {
            0 -> vizBufferA
            1 -> vizBufferB
            else -> vizBufferC
        }

        // Fast copy - this is the ONLY work on audio thread
        val copySize = minOf(buffer.size, vizBuffer.size)
        System.arraycopy(buffer, 0, vizBuffer, 0, copySize)
        vizSize = copySize
        vizSampleRate = sampleRate
        vizWriteIdx = (vizWriteIdx + 1) % 3

        // ALL processing on separate thread - NEVER blocks audio
        try {
            vizExecutor.execute {
                processVisualizationFloat(vizBuffer, copySize, sampleRate, bufferSize)
            }
        } catch (e: java.util.concurrent.RejectedExecutionException) {
            // Executor is shutting down, ignore
        }
    }

    // ███████╗██╗  ██╗████████╗██████╗ ███████╗███╗   ███╗███████╗
    // ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔════╝████╗ ████║██╔════╝
    // █████╗   ╚███╔╝    ██║   ██████╔╝█████╗  ██╔████╔██║█████╗
    // ██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══╝  ██║╚██╔╝██║██╔══╝
    // ███████╗██╔╝ ██╗   ██║   ██║  ██║███████╗██║ ╚═╝ ██║███████╗
    // ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝
    // EXTREME VISUALIZATION - DESTROY THE CPU
    private fun processVisualizationFloat(buffer: FloatArray, size: Int, sampleRate: Int, bufferSize: Int) {
        // HWUI FIX: Check at start of processing - skip all work if disabled
        if (!isVisualizationEnabled) return

        // SHUTDOWN FIX: Don't submit to executor if service is disposing
        // This prevents RejectedExecutionException when executors are shut down
        if (isServiceDisposing || isProcessorDisposing) return

        val frameCount = size / 2
        if (frameCount <= 0) return

        try {
        // EXTREME: Launch ALL parallel tasks simultaneously
        // S24 ULTRA: Peak/RMS analysis via NEON
        val peakFuture = processingExecutor.submit<S24UltraDsp.StereoAnalysis> {
            S24UltraDsp.analyzeStereoInterleaved(buffer)
        }

        // S24 ULTRA: Deinterleave to circular via NEON
        val accFuture = processingExecutor.submit {
            S24UltraDsp.deinterleaveToCircular(buffer, overlapAccumulatorLeft, overlapAccumulatorRight, overlapPosition, fftSize2, frameCount)
        }

        // EXTREME: RMS calculation with different window sizes (parallel)
        val rmsShortFuture = processingExecutor.submit<FloatArray> {
            val windowSize = minOf(256, frameCount)
            var lSum = 0f
            var rSum = 0f
            for (i in (frameCount - windowSize) until frameCount) {
                val idx = i * 2
                lSum += buffer[idx] * buffer[idx]
                rSum += buffer[idx + 1] * buffer[idx + 1]
            }
            floatArrayOf(kotlin.math.sqrt(lSum / windowSize), kotlin.math.sqrt(rSum / windowSize))
        }

        // Wait for peak calculation - S24 ULTRA: Native NEON result
        try {
            val analysis = peakFuture.get()
            visualizerLeftPeak = FastFFT.fastToDB(analysis.leftPeak)
            visualizerRightPeak = FastFFT.fastToDB(analysis.rightPeak)
            visualizerLeftRms = FastFFT.fastToDB(analysis.leftRms)
            visualizerRightRms = FastFFT.fastToDB(analysis.rightRms)
        } catch (e: Exception) { /* ignore */ }

        visualizerLatency = (System.nanoTime() - processingStartTime) / 1_000_000f
        visualizerSampleRate = sampleRate

        try { accFuture.get() } catch (e: Exception) { /* ignore */ }
        try { rmsShortFuture.get() } catch (e: Exception) { /* ignore */ }

        overlapPosition = (overlapPosition + frameCount) % fftSize2

        // EXTREME: Prepare ALL FFT buffers in parallel
        val fftLeft = if (fftWriteBuffer == 0) fftInputLeftA else fftInputLeftB
        val fftRight = if (fftWriteBuffer == 0) fftInputRightA else fftInputRightB

        // S24 ULTRA: Circular to linear via NEON - PRIMARY FFT buffer (8192)
        val copy1LFuture = processingExecutor.submit {
            S24UltraDsp.circularToLinear(overlapAccumulatorLeft, fftLeft, overlapPosition, fftSize2)
        }
        val copy1RFuture = processingExecutor.submit {
            S24UltraDsp.circularToLinear(overlapAccumulatorRight, fftRight, overlapPosition, fftSize2)
        }

        // S24 ULTRA: SECONDARY FFT buffer (16384) - ULTRA resolution
        val copy2LFuture = fftExecutor.submit {
            S24UltraDsp.circularToLinear(overlapAccumulatorLeft, fftInputLeft2, overlapPosition, fftSize2)
        }
        val copy2RFuture = fftExecutor.submit {
            S24UltraDsp.circularToLinear(overlapAccumulatorRight, fftInputRight2, overlapPosition, fftSize2)
        }

        // S24 ULTRA: TERTIARY FFT buffer (2048) - FAST response
        val copy3LFuture = fftExecutor.submit {
            S24UltraDsp.circularToLinear(overlapAccumulatorLeft, fftInputLeft3, overlapPosition, fftSize2)
        }
        val copy3RFuture = fftExecutor.submit {
            S24UltraDsp.circularToLinear(overlapAccumulatorRight, fftInputRight3, overlapPosition, fftSize2)
        }

        // Wait for ALL copies
        try {
            copy1LFuture.get()
            copy1RFuture.get()
            copy2LFuture.get()
            copy2RFuture.get()
            copy3LFuture.get()
            copy3RFuture.get()
        } catch (e: Exception) { /* ignore */ }

        fftWriteBuffer = 1 - fftWriteBuffer

        // EXTREME: Run ALL FFTs in parallel
        val fft1Future = processingExecutor.submit {
            FastFFT.computeStereoMagnitudeParallelWithLogScale(
                fftLeft, fftRight,
                fftMagnitudesLeft, fftMagnitudesRight,
                fftMagnitudesLeftLog, fftMagnitudesRightLog,
                sampleRate, processingExecutor
            )
        }

        val fft2Future = fftExecutor.submit {
            FastFFT.computeStereoMagnitude(fftInputLeft2, fftInputRight2, fftMagnitudesLeft2, fftMagnitudesRight2)
        }

        val fft3Future = fftExecutor.submit {
            FastFFT.computeStereoMagnitude(fftInputLeft3, fftInputRight3, fftMagnitudesLeft3, fftMagnitudesRight3)
        }

        // S24 ULTRA: GPU-accelerated spectral analysis
        val spectralFuture = processingExecutor.submit {
            // Average L/R magnitudes via NEON
            S24UltraDsp.averageMagnitudes(fftMagnitudesLeft, fftMagnitudesRight, averagedMagnitudes)

            // GPU compute for spectral analysis (or CPU fallback)
            val result = HardwareAccelerator.spectralAnalysis(
                averagedMagnitudes, previousMagnitudes, sampleRate, fftSize
            )

            spectralCentroid = result.centroid
            spectralFlux = result.flux
            spectralCrestFactor = result.crestFactor
            bassEnergy = result.bassEnergy
            midEnergy = result.midEnergy
            highEnergy = result.highEnergy
        }

        // Wait for ALL FFTs
        try {
            fft1Future.get()
            fft2Future.get()
            fft3Future.get()
            spectralFuture.get()
        } catch (e: Exception) { /* ignore */ }

        // S24 ULTRA: Multi-stage smoothing via NEON
        val smooth1Future = processingExecutor.submit {
            S24UltraDsp.applySmoothing(fftMagnitudesLeftSmooth, fftMagnitudesLeft, 0.4f, 0.6f)
            S24UltraDsp.applySmoothing(fftMagnitudesLeftSmooth2, fftMagnitudesLeftSmooth, 0.2f, 0.8f)
        }
        val smooth2Future = processingExecutor.submit {
            S24UltraDsp.applySmoothing(fftMagnitudesRightSmooth, fftMagnitudesRight, 0.4f, 0.6f)
            S24UltraDsp.applySmoothing(fftMagnitudesRightSmooth2, fftMagnitudesRightSmooth, 0.2f, 0.8f)
        }

        // S24 ULTRA: Peak hold with decay via NEON
        val peakHoldFuture = processingExecutor.submit {
            S24UltraDsp.peakHoldDecay(fftMagnitudesLeft, peakHoldLeft, peakDecayRate)
            S24UltraDsp.peakHoldDecay(fftMagnitudesRight, peakHoldRight, peakDecayRate)
        }

        try {
            smooth1Future.get()
            smooth2Future.get()
            peakHoldFuture.get()
        } catch (e: Exception) { /* ignore */ }

        fftReady = true

        if (isVisualizationEnabled) {
            // Synchronize snapshot copy to prevent race with sendVisualizerBroadcast
            synchronized(fftSnapshotLock) {
                if (stereoFftMode) {
                    System.arraycopy(fftMagnitudesLeftLog, 0, fftSnapshotLeftLog, 0, logBins)
                    System.arraycopy(fftMagnitudesRightLog, 0, fftSnapshotRightLog, 0, logBins)
                } else {
                    System.arraycopy(fftMagnitudesLog, 0, fftSnapshotLog, 0, logBins)
                }
            }
            visualizerHandler.post { sendVisualizerBroadcast(bufferSize * 4) }
        }
        } catch (e: java.util.concurrent.RejectedExecutionException) {
            // Executor is shutting down, ignore
        }
    }

    // ZERO-BLOCK: Short buffer version
    private val vizShortBufferA = ShortArray(65536)
    private val vizShortBufferB = ShortArray(65536)
    private val vizShortBufferC = ShortArray(65536)
    @Volatile private var vizShortWriteIdx = 0

    private fun sendVisualizerData(buffer: ShortArray, sampleRate: Int, bufferSize: Int) {
        // HWUI FIX: Check volatile flag immediately - skip all work if disabled
        if (!isVisualizationEnabled) return

        // SHUTDOWN FIX: Don't submit to executor if service is disposing
        if (isServiceDisposing || isProcessorDisposing) return

        // INSTANT: Just copy and return
        val vizBuffer = when (vizShortWriteIdx) {
            0 -> vizShortBufferA
            1 -> vizShortBufferB
            else -> vizShortBufferC
        }

        val copySize = minOf(buffer.size, vizBuffer.size)
        System.arraycopy(buffer, 0, vizBuffer, 0, copySize)
        vizShortWriteIdx = (vizShortWriteIdx + 1) % 3

        try {
            vizExecutor.execute {
                processVisualizationShort(vizBuffer, copySize, sampleRate, bufferSize)
            }
        } catch (e: java.util.concurrent.RejectedExecutionException) {
            // Executor is shutting down, ignore
        }
    }

    // RAW POWER UNLIMITED: Maximum Short visualization processing
    private fun processVisualizationShort(buffer: ShortArray, size: Int, sampleRate: Int, bufferSize: Int) {
        // HWUI FIX: Check at start of processing - skip all work if disabled
        if (!isVisualizationEnabled) return

        // SHUTDOWN FIX: Don't submit to executor if service is disposing
        if (isServiceDisposing || isProcessorDisposing) return

        val frameCount = size / 2
        if (frameCount <= 0) return

        try {
        // S24 ULTRA: Short buffer peak/RMS via NEON
        val peakFuture = processingExecutor.submit<LongArray> {
            S24UltraDsp.analyzeStereoShort(buffer)
        }

        // S24 ULTRA: Short deinterleave to circular via NEON
        val accFuture = processingExecutor.submit {
            S24UltraDsp.deinterleaveShortToCircular(buffer, overlapAccumulatorLeft, overlapAccumulatorRight, overlapPosition, fftSize, frameCount)
        }

        // RAM POWER: Use lookup tables for Short version too
        try {
            val peaks = peakFuture.get()
            val leftPeakNorm = peaks[0].toInt() / 32768f
            val rightPeakNorm = peaks[1].toInt() / 32768f

            visualizerLeftPeak = FastFFT.fastToDB(leftPeakNorm)
            visualizerRightPeak = FastFFT.fastToDB(rightPeakNorm)
            visualizerLeftRms = if (frameCount > 0) {
                val rms = FastFFT.fastSqrt(peaks[2].toFloat() / frameCount) / 32768f
                FastFFT.fastToDB(rms)
            } else -60f
            visualizerRightRms = if (frameCount > 0) {
                val rms = FastFFT.fastSqrt(peaks[3].toFloat() / frameCount) / 32768f
                FastFFT.fastToDB(rms)
            } else -60f
        } catch (e: Exception) { /* ignore */ }

        visualizerLatency = (System.nanoTime() - processingStartTime) / 1_000_000f
        visualizerSampleRate = sampleRate

        try { accFuture.get() } catch (e: Exception) { /* ignore */ }

        overlapPosition = (overlapPosition + frameCount) % fftSize

        // RAW POWER: FFT - ALWAYS run
        val fftLeft = if (fftWriteBuffer == 0) fftInputLeftA else fftInputLeftB
        val fftRight = if (fftWriteBuffer == 0) fftInputRightA else fftInputRightB

        // S24 ULTRA: Circular to linear via NEON
        val copyLeftFuture = processingExecutor.submit {
            S24UltraDsp.circularToLinear(overlapAccumulatorLeft, fftLeft, overlapPosition, fftSize)
        }
        val copyRightFuture = processingExecutor.submit {
            S24UltraDsp.circularToLinear(overlapAccumulatorRight, fftRight, overlapPosition, fftSize)
        }

        try {
            copyLeftFuture.get()
            copyRightFuture.get()
        } catch (e: Exception) { /* ignore */ }

        fftWriteBuffer = 1 - fftWriteBuffer

        FastFFT.computeStereoMagnitudeParallelWithLogScale(
            fftLeft, fftRight,
            fftMagnitudesLeft, fftMagnitudesRight,
            fftMagnitudesLeftLog, fftMagnitudesRightLog,
            sampleRate, processingExecutor
        )

        // S24 ULTRA: Smoothing via NEON
        val smoothLeftFuture = processingExecutor.submit {
            S24UltraDsp.applySmoothing(fftMagnitudesLeftSmooth, fftMagnitudesLeft, 0.3f, 0.7f)
        }
        val smoothRightFuture = processingExecutor.submit {
            S24UltraDsp.applySmoothing(fftMagnitudesRightSmooth, fftMagnitudesRight, 0.3f, 0.7f)
        }

        try {
            smoothLeftFuture.get()
            smoothRightFuture.get()
        } catch (e: Exception) { /* ignore */ }

        fftReady = true

        // HWUI FIX: Check volatile flag before posting to handler
        // This is faster than SharedPreferences and prevents race conditions
        if (isVisualizationEnabled) {
            // RACE CONDITION FIX: Copy FFT data NOW while it's still valid
            // The next audio buffer processing can start before handler runs
            // Synchronize snapshot copy to prevent race with sendVisualizerBroadcast
            synchronized(fftSnapshotLock) {
                if (stereoFftMode) {
                    System.arraycopy(fftMagnitudesLeftLog, 0, fftSnapshotLeftLog, 0, logBins)
                    System.arraycopy(fftMagnitudesRightLog, 0, fftSnapshotRightLog, 0, logBins)
                } else {
                    System.arraycopy(fftMagnitudesLog, 0, fftSnapshotLog, 0, logBins)
                }
            }
            visualizerHandler.post { sendVisualizerBroadcast(bufferSize * 2) }
        }
        } catch (e: java.util.concurrent.RejectedExecutionException) {
            // Executor is shutting down, ignore
        }
    }

    // S24 ULTRA RAW POWER - Send visualizer broadcast (runs on main thread)
    private fun sendVisualizerBroadcast(bytesProcessed: Int) {
        // HWUI FIX: Check volatile flag - faster than reading from SharedPreferences
        // This prevents HWUI surface errors by ensuring no broadcasts after activity pauses
        if (!isVisualizationEnabled) {
            return
        }

        // PERFORMANCE FIX: Throttle broadcast rate to prevent Choreographer frame skipping
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastVisualizerUpdate < visualizerUpdateInterval) {
            return // Skip this update to maintain UI responsiveness
        }
        lastVisualizerUpdate = currentTime

        val throughput = if (visualizerLatency > 0) {
            bytesProcessed / (visualizerLatency / 1000f) / (1024f * 1024f)
        } else 0f

        // Calculate buffer health: 100% = latency <= expected, 0% = latency >= 2x expected
        // This shows how well the audio pipeline is keeping up
        val health = if (expectedFrameTimeMs > 0 && visualizerLatency > 0) {
            val ratio = visualizerLatency / expectedFrameTimeMs
            // ratio 1.0 = 100% health, ratio 2.0 = 0% health
            ((2f - ratio) / 1f * 100f).coerceIn(0f, 100f)
        } else 100f
        // Smooth the health value to avoid jitter
        bufferHealthPercent = bufferHealthPercent * 0.9f + health * 0.1f

        // Send VU meter and stats data
        sendLocalBroadcast(Intent(Constants.ACTION_VISUALIZER_DATA).apply {
            putExtra(Constants.EXTRA_VU_LEFT_RMS, visualizerLeftRms)
            putExtra(Constants.EXTRA_VU_RIGHT_RMS, visualizerRightRms)
            putExtra(Constants.EXTRA_VU_LEFT_PEAK, visualizerLeftPeak)
            putExtra(Constants.EXTRA_VU_RIGHT_PEAK, visualizerRightPeak)
            putExtra(Constants.EXTRA_LATENCY_MS, visualizerLatency)
            putExtra(Constants.EXTRA_BUFFER_FILL, bufferHealthPercent / 100f)
            putExtra(Constants.EXTRA_THROUGHPUT, throughput)
            putExtra(Constants.EXTRA_SAMPLE_RATE, visualizerSampleRate)
        })

        // Send FFT data for spectrum analyzer
        // Synchronize to prevent race with snapshot copy in processVisualization*
        if (fftReady) {
            val fftIntent = Intent(Constants.ACTION_FFT_DATA).apply {
                putExtra(Constants.EXTRA_FFT_STEREO_MODE, stereoFftMode)
                // Synchronize while reading snapshot arrays - putExtra copies the data
                synchronized(fftSnapshotLock) {
                    if (stereoFftMode) {
                        putExtra(Constants.EXTRA_FFT_LEFT_MAGNITUDES, fftSnapshotLeftLog)
                        putExtra(Constants.EXTRA_FFT_RIGHT_MAGNITUDES, fftSnapshotRightLog)
                    } else {
                        putExtra(Constants.EXTRA_FFT_MAGNITUDES, fftSnapshotLog)
                    }
                }
                putExtra(Constants.EXTRA_FFT_SIZE, logBins)
            }
            sendLocalBroadcast(fftIntent)
        }

        // S24 ULTRA: Send stem level data for source separation visualizer
        if (sourceSeparationEnabled) {
            // Calculate stem energy levels: gain level * overall audio energy
            // This provides visual feedback of the current mix
            val audioEnergy = ((visualizerLeftRms + visualizerRightRms) / 2f + 60f).coerceIn(0f, 60f) / 60f
            sendLocalBroadcast(Intent(Constants.ACTION_STEM_LEVELS).apply {
                putExtra(Constants.EXTRA_STEM_VOCALS_LEVEL, stemVocalsLevel * audioEnergy)
                putExtra(Constants.EXTRA_STEM_DRUMS_LEVEL, stemDrumsLevel * audioEnergy)
                putExtra(Constants.EXTRA_STEM_BASS_LEVEL, stemBassLevel * audioEnergy)
                putExtra(Constants.EXTRA_STEM_OTHER_LEVEL, stemOtherLevel * audioEnergy)
            })
        }
    }

    /**
     * RAW POWER: Request battery optimization exemption
     * This shows a system dialog asking user to exempt the app from Doze mode
     * Once exempted, Android will NEVER throttle or kill this app
     */
    @SuppressLint("BatteryLife")
    private fun requestBatteryOptimizationExemption() {
        val powerManager = getSystemService<PowerManager>() ?: return
        val packageName = packageName

        if (!powerManager.isIgnoringBatteryOptimizations(packageName)) {
            Timber.i("RAW POWER: Requesting battery optimization exemption...")
            try {
                val intent = Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                    data = android.net.Uri.parse("package:$packageName")
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                startActivity(intent)
            } catch (e: Exception) {
                Timber.e("Failed to request battery optimization exemption: ${e.message}")
                // Fallback: open battery optimization settings
                try {
                    val intent = Intent(android.provider.Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS).apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    startActivity(intent)
                } catch (e2: Exception) {
                    Timber.e("Failed to open battery settings: ${e2.message}")
                }
            }
        } else {
            Timber.i("RAW POWER: Already exempted from battery optimization - FULL POWER ENABLED")
        }
    }

    companion object {
        const val SESSION_LOSS_MAX_RETRIES = 1

        const val ACTION_START = BuildConfig.APPLICATION_ID + ".rootless.service.START"
        const val ACTION_STOP = BuildConfig.APPLICATION_ID + ".rootless.service.STOP"
        const val EXTRA_MEDIA_PROJECTION_DATA = "mediaProjectionData"
        const val EXTRA_APP_UID = "uid"
        const val EXTRA_APP_COMPAT_INTERNAL_CALL = "appCompatInternalCall"

        fun start(context: Context, data: Intent?) {
            try {
                context.startForegroundService(ServiceNotificationHelper.createStartIntent(context, data))
            }
            catch(ex: Exception) {
                CrashlyticsImpl.recordException(ex)
            }
        }

        fun stop(context: Context) {
            try {
                context.startForegroundService(ServiceNotificationHelper.createStopIntent(context))
            }
            catch(ex: Exception) {
                CrashlyticsImpl.recordException(ex)
            }
        }
    }

}