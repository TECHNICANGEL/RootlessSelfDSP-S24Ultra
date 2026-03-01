package me.timschneeberger.rootlessjamesdsp.audio

import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import me.timschneeberger.rootlessjamesdsp.utils.Constants
import timber.log.Timber
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * S24 ULTRA - Unified Hardware Accelerator
 *
 * Available acceleration:
 * - NEON SIMD (ARM64) - Always available, primary DSP
 * - GPU Compute (Adreno 750) - For large parallel operations
 * - NNAPI (NPU) - Source separation, spatial audio
 */
object HardwareAccelerator {

    private const val TAG = "HardwareAccelerator"

    private val npuAvailable = AtomicBoolean(false)
    private val gpuAvailable = AtomicBoolean(false)
    private val sourceSepAvailable = AtomicBoolean(false)
    private val spatialAvailable = AtomicBoolean(false)
    private val initialized = AtomicBoolean(false)
    private val initializationComplete = AtomicBoolean(false)

    private var sourceSeparatorBackend = AtomicReference("Unavailable")
    private var sourceSeparator: SourceSeparator? = null
    private var spatialAudio: SpatialAudio? = null
    private var contentAnalyzer: ContentAnalyzer? = null
    private var parallelExecutor: ExecutorService? = null
    private var appContext: Context? = null

    // Main thread handler for UI updates
    private val mainHandler = Handler(Looper.getMainLooper())

    // State listener for broadcasting separation state to UI
    // Posts to main thread to ensure immediate UI updates
    private val separationStateListener = object : SourceSeparator.StateListener {
        override fun onStateChanged(state: SourceSeparator.SeparationState, progress: Float, message: String, npuUsage: Float) {
            appContext?.let { ctx ->
                mainHandler.post {
                    val intent = Intent(Constants.ACTION_SEPARATION_STATE).apply {
                        putExtra(Constants.EXTRA_SEPARATION_STATE, state.ordinal)
                        putExtra(Constants.EXTRA_SEPARATION_PROGRESS, progress)
                        putExtra(Constants.EXTRA_SEPARATION_MESSAGE, message)
                        putExtra(Constants.EXTRA_NPU_USAGE, npuUsage)
                    }
                    LocalBroadcastManager.getInstance(ctx).sendBroadcast(intent)
                }
            }
        }

        override fun onPipelineStats(bufferPct: Int, stftMs: Int, npuMs: Int, istftMs: Int, queueSize: Int, state: SourceSeparator.SeparationState, progress: Float) {
            appContext?.let { ctx ->
                mainHandler.post {
                    val intent = Intent(Constants.ACTION_PIPELINE_STATS).apply {
                        putExtra(Constants.EXTRA_BUFFER_PCT, bufferPct)
                        putExtra(Constants.EXTRA_STFT_MS, stftMs)
                        putExtra(Constants.EXTRA_NPU_MS, npuMs)
                        putExtra(Constants.EXTRA_ISTFT_MS, istftMs)
                        putExtra(Constants.EXTRA_QUEUE_SIZE, queueSize)
                        putExtra(Constants.EXTRA_SEPARATION_STATE, state.ordinal)
                        putExtra(Constants.EXTRA_SEPARATION_PROGRESS, progress)
                    }
                    LocalBroadcastManager.getInstance(ctx).sendBroadcast(intent)
                }
            }
        }

        override fun onModelStats(vocalsMs: Int, drumsMs: Int, bassMs: Int, otherMs: Int, totalMs: Int, usingNpu: Boolean, backendName: String) {
            appContext?.let { ctx ->
                mainHandler.post {
                    val intent = Intent(Constants.ACTION_MODEL_STATS).apply {
                        putExtra(Constants.EXTRA_VOCALS_MS, vocalsMs)
                        putExtra(Constants.EXTRA_DRUMS_MS, drumsMs)
                        putExtra(Constants.EXTRA_BASS_MS, bassMs)
                        putExtra(Constants.EXTRA_OTHER_MS, otherMs)
                        putExtra(Constants.EXTRA_TOTAL_MS, totalMs)
                        putExtra(Constants.EXTRA_USING_NPU, usingNpu)
                        putExtra(Constants.EXTRA_BACKEND_NAME, backendName)
                    }
                    LocalBroadcastManager.getInstance(ctx).sendBroadcast(intent)
                }
            }
        }
    }

    private var cpuProcessingTimeNs = 0L
    private var npuProcessingTimeNs = 0L
    private var gpuProcessingTimeNs = 0L
    private var totalOperations = 0L

    private val processingMode = AtomicReference(ProcessingMode.AUTO)

    enum class ProcessingMode { AUTO, CPU_ONLY, GPU_PREFER, NPU_PREFER }

    fun initialize(context: Context): Boolean {
        if (initialized.getAndSet(true)) {
            return npuAvailable.get() || gpuAvailable.get()
        }

        Timber.tag(TAG).i("Initializing hardware accelerators...")

        parallelExecutor = Executors.newFixedThreadPool(4) { r ->
            Thread(r, "HW-Accel").apply { priority = Thread.MAX_PRIORITY - 1 }
        }

        // Initialize native DSP (NEON)
        S24UltraDsp.initLookupTables()
        FastFFT.preWarm()

        initializeNpu(context)
        initializeGpu()
        logCapabilities()

        initializationComplete.set(true)
        return npuAvailable.get() || gpuAvailable.get()
    }

    private fun initializeNpu(context: Context) {
        // Check NNAPI support
        npuAvailable.set(NpuEngine.isSupported())
        if (npuAvailable.get()) {
            Timber.tag(TAG).i("NNAPI available:\n${NpuEngine.getDeviceInfo()}")
        } else {
            // Not having NNAPI is normal on many devices - use debug level
            Timber.tag(TAG).d("NNAPI not available - this is normal on many devices")
        }

        // Store context for state broadcasts
        appContext = context.applicationContext

        // Initialize Source Separator (Demucs)
        try {
            sourceSeparator = SourceSeparator(context)
            sourceSeparator!!.setStateListener(separationStateListener)
            if (sourceSeparator!!.initialize()) {
                sourceSepAvailable.set(true)
                val backend = sourceSeparator!!.getBackendName()
                sourceSeparatorBackend.set(backend)
                Timber.tag(TAG).i("Source Separation enabled (backend: $backend)")
            } else {
                sourceSeparatorBackend.set("Model Not Found")
                Timber.tag(TAG).d("Source Separation not available (model may not be present)")
            }
        } catch (e: Exception) {
            sourceSeparatorBackend.set("Init Failed")
            Timber.tag(TAG).d("Source Separator init skipped: ${e.message}")
        }

        // Initialize Spatial Audio (HRTF)
        try {
            spatialAudio = SpatialAudio(context)
            if (spatialAudio!!.initialize()) {
                spatialAvailable.set(true)
                Timber.tag(TAG).i("Spatial Audio enabled (HRTF + Head Tracking)")
            }
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "Spatial Audio init failed")
        }

        // Initialize Content Analyzer (for auto-presets)
        try {
            contentAnalyzer = ContentAnalyzer(48000)
            Timber.tag(TAG).i("Content Analyzer enabled")
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "Content Analyzer init failed")
        }
    }

    private fun initializeGpu() {
        try {
            if (GpuCompute.initialize()) {
                gpuAvailable.set(true)
                Timber.tag(TAG).i("GPU Compute enabled (Adreno 750)")
            }
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "GPU Compute initialization failed")
        }
    }

    private fun logCapabilities() {
        Timber.tag(TAG).i("""
            Hardware Accelerator Status:
            - NEON SIMD: ENABLED (Cortex-X4, always available)
            - GPU Compute: ${if (gpuAvailable.get()) "ENABLED (Adreno 750)" else "DISABLED"}
            - NNAPI: ${if (npuAvailable.get()) "ENABLED" else "DISABLED"}
            - Source Separation: ${if (sourceSepAvailable.get()) "ENABLED (Demucs)" else "DISABLED (no model)"}
            - Spatial Audio: ${if (spatialAvailable.get()) "ENABLED (HRTF)" else "DISABLED"}
            - CPU cores: ${Runtime.getRuntime().availableProcessors()}
        """.trimIndent())
    }

    fun setProcessingMode(mode: ProcessingMode) {
        processingMode.set(mode)
    }

    // ═══════════════════════════════════════════════════════════════
    // FFT - S24UltraDsp native (NEON)
    // PERFORMANCE: Using AudioBufferPool to avoid per-call allocations
    // ═══════════════════════════════════════════════════════════════

    // PERFORMANCE: Cache buffer pools for common FFT sizes
    private var lastMagnitudePoolSize = 0
    private var magnitudePool: AudioBufferPool.FloatBufferPool? = null
    private var lastResultPoolSize = 0
    private var resultPool: AudioBufferPool.FloatBufferPool? = null

    fun fft(input: FloatArray, size: Int = input.size): FloatArray {
        // Edge case: need at least 2 samples for FFT
        if (input.size < 2) return FloatArray(0)

        val startTime = System.nanoTime()

        // PERFORMANCE: Use buffer pools to avoid allocations
        val magnitudeSize = input.size / 2
        if (magnitudeSize != lastMagnitudePoolSize) {
            magnitudePool = AudioBufferPool.getFloatPool(magnitudeSize, 4)
            lastMagnitudePoolSize = magnitudeSize
        }
        val resultSize = magnitudeSize * 2
        if (resultSize != lastResultPoolSize) {
            resultPool = AudioBufferPool.getFloatPool(resultSize, 4)
            lastResultPoolSize = resultSize
        }

        val magnitudes = magnitudePool?.acquire() ?: FloatArray(magnitudeSize)
        S24UltraDsp.fftMagnitude(input, magnitudes)
        cpuProcessingTimeNs += System.nanoTime() - startTime
        totalOperations++

        // Return as interleaved format for compatibility
        val result = resultPool?.acquire() ?: FloatArray(resultSize)
        S24UltraDsp.magnitudeToInterleaved(magnitudes, result)

        // Release magnitude buffer back to pool
        magnitudePool?.release(magnitudes)

        return result
    }

    /**
     * Release an FFT result buffer back to the pool
     * Call this after you're done using the result from fft()
     */
    fun releaseFFTBuffer(buffer: FloatArray) {
        resultPool?.release(buffer)
    }

    fun fftMagnitude(input: FloatArray, size: Int = input.size): FloatArray {
        // Edge case: need at least 2 samples for FFT
        if (input.size < 2) return FloatArray(0)

        val magnitudeSize = input.size / 2
        if (magnitudeSize != lastMagnitudePoolSize) {
            magnitudePool = AudioBufferPool.getFloatPool(magnitudeSize, 4)
            lastMagnitudePoolSize = magnitudeSize
        }

        val magnitudes = magnitudePool?.acquire() ?: FloatArray(magnitudeSize)
        S24UltraDsp.fftMagnitude(input, magnitudes)
        return magnitudes
    }

    /**
     * Release an fftMagnitude result buffer back to the pool
     */
    fun releaseMagnitudeBuffer(buffer: FloatArray) {
        magnitudePool?.release(buffer)
    }

    // ═══════════════════════════════════════════════════════════════
    // Dynamics - S24UltraDsp native (NEON)
    // ═══════════════════════════════════════════════════════════════

    fun compress(
        samples: FloatArray,
        threshold: Float = -20f,
        ratio: Float = 4f,
        attack: Float = 0.01f,
        release: Float = 0.1f,
        makeupDb: Float = 0f,
        sampleRate: Int = 48000
    ): FloatArray {
        val output = samples.copyOf()
        S24UltraDsp.compressor(output, threshold, ratio, attack * 1000f, release * 1000f, makeupDb, sampleRate)
        return output
    }

    fun limit(
        samples: FloatArray,
        threshold: Float = -0.1f,
        release: Float = 0.05f,
        sampleRate: Int = 48000
    ): FloatArray {
        val output = samples.copyOf()
        S24UltraDsp.limiter(output, threshold, release * 1000f, sampleRate)
        return output
    }

    fun gate(
        samples: FloatArray,
        threshold: Float = -40f,
        attack: Float = 0.001f,
        hold: Float = 0.05f,
        release: Float = 0.1f,
        range: Float = -80f,
        sampleRate: Int = 48000
    ): FloatArray {
        val output = samples.copyOf()
        S24UltraDsp.gate(output, threshold, attack * 1000f, hold * 1000f, release * 1000f, range, sampleRate)
        return output
    }

    // ═══════════════════════════════════════════════════════════════
    // Stereo - S24UltraDsp native (NEON)
    // ═══════════════════════════════════════════════════════════════

    fun stereoWidth(left: FloatArray, right: FloatArray, width: Float = 1f): Pair<FloatArray, FloatArray> {
        val outL = left.copyOf()
        val outR = right.copyOf()
        S24UltraDsp.stereoWidth(outL, outR, width)
        return Pair(outL, outR)
    }

    fun crossfeed(left: FloatArray, right: FloatArray, amount: Float = 0.3f): Pair<FloatArray, FloatArray> {
        val outL = left.copyOf()
        val outR = right.copyOf()
        S24UltraDsp.crossfeed(outL, outR, amount)
        return Pair(outL, outR)
    }

    // ═══════════════════════════════════════════════════════════════
    // Analysis - S24UltraDsp native (NEON)
    // ═══════════════════════════════════════════════════════════════

    fun findPeak(samples: FloatArray): Float = S24UltraDsp.findPeak(samples)

    fun calculateRms(samples: FloatArray): Float = S24UltraDsp.calculateRms(samples)

    // ═══════════════════════════════════════════════════════════════
    // GPU Compute - Parallel spectral analysis (Adreno 750)
    // ═══════════════════════════════════════════════════════════════

    fun spectralAnalysis(
        magnitudes: FloatArray,
        previousMagnitudes: FloatArray,
        sampleRate: Int,
        fftSize: Int
    ): GpuCompute.SpectralResult {
        if (gpuAvailable.get() && magnitudes.size >= 2048 && processingMode.get() != ProcessingMode.CPU_ONLY) {
            val startTime = System.nanoTime()
            val result = GpuCompute.spectralAnalysis(magnitudes, previousMagnitudes, sampleRate, fftSize)
            gpuProcessingTimeNs += System.nanoTime() - startTime
            totalOperations++
            return result
        }
        // CPU fallback
        return GpuCompute.spectralAnalysis(magnitudes, previousMagnitudes, sampleRate, fftSize)
    }

    fun gpuSmoothing(current: FloatArray, target: FloatArray, attack: Float, release: Float) {
        if (gpuAvailable.get() && current.size >= 1024 && processingMode.get() != ProcessingMode.CPU_ONLY) {
            val startTime = System.nanoTime()
            GpuCompute.smoothing(current, target, attack, release)
            gpuProcessingTimeNs += System.nanoTime() - startTime
            totalOperations++
        } else {
            S24UltraDsp.applySmoothing(current, target, attack, release)
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Source Separation (Demucs) - NNAPI/NPU
    // ═══════════════════════════════════════════════════════════════

    fun separateSources(input: FloatArray): SourceSeparator.StemResult? {
        if (!sourceSepAvailable.get()) return null
        val startTime = System.nanoTime()
        val result = sourceSeparator?.separate(input)
        npuProcessingTimeNs += System.nanoTime() - startTime
        totalOperations++
        return result
    }

    fun processWithStemLevels(input: FloatArray): FloatArray {
        Timber.tag(TAG).w("▶▶▶ processWithStemLevels() called, inputSize=${input.size}")
        Timber.tag(TAG).w("  sourceSepAvailable=${sourceSepAvailable.get()}, separator=${sourceSeparator != null}")

        if (!sourceSepAvailable.get()) {
            Timber.tag(TAG).w("⚠️ Source separation not available, returning input unchanged")
            return input
        }
        val startTime = System.nanoTime()
        Timber.tag(TAG).w("🔄 Calling sourceSeparator.process()...")
        val result = sourceSeparator?.process(input) ?: input
        val elapsed = (System.nanoTime() - startTime) / 1_000_000
        npuProcessingTimeNs += System.nanoTime() - startTime
        totalOperations++
        Timber.tag(TAG).w("✅ processWithStemLevels complete in ${elapsed}ms, resultSize=${result.size}")
        return result
    }

    // Stem controls
    fun setVocalsLevel(level: Float) = sourceSeparator?.setVocalsLevel(level)
    fun setDrumsLevel(level: Float) = sourceSeparator?.setDrumsLevel(level)
    fun setBassLevel(level: Float) = sourceSeparator?.setBassLevel(level)
    fun setOtherLevel(level: Float) = sourceSeparator?.setOtherLevel(level)

    fun setKaraokeMode() = sourceSeparator?.setKaraokeMode()
    fun setVocalsOnlyMode() = sourceSeparator?.setVocalsOnlyMode()
    fun resetStemLevels() = sourceSeparator?.resetLevels()

    /**
     * Reset source separator filter states
     * Call when audio playback restarts to prevent filter "ringing"
     */
    fun resetSourceSeparatorFilters() = sourceSeparator?.resetFilters()

    // ═══════════════════════════════════════════════════════════════
    // Spatial Audio (HRTF + Head Tracking)
    // ═══════════════════════════════════════════════════════════════

    fun processSpatial(left: FloatArray, right: FloatArray): Pair<FloatArray, FloatArray> {
        if (!spatialAvailable.get()) return Pair(left, right)
        val startTime = System.nanoTime()
        val result = spatialAudio?.process(left, right) ?: Pair(left, right)
        cpuProcessingTimeNs += System.nanoTime() - startTime  // HRTF uses NEON
        totalOperations++
        return result
    }

    fun processSpatialInterleaved(stereo: FloatArray): FloatArray {
        if (!spatialAvailable.get()) return stereo
        return spatialAudio?.processInterleaved(stereo) ?: stereo
    }

    fun enableHeadTracking() = spatialAudio?.enableHeadTracking()
    fun disableHeadTracking() = spatialAudio?.disableHeadTracking()
    fun recenterHeadTracking() = spatialAudio?.recenter()
    fun isHeadTrackingEnabled() = spatialAudio?.isHeadTrackingEnabled() ?: false
    fun getHeadAzimuth() = spatialAudio?.getAzimuth() ?: 0f
    fun getHeadElevation() = spatialAudio?.getElevation() ?: 0f
    fun setSpatialIntensity(intensity: Float) = spatialAudio?.setIntensity(intensity)
    fun getSpatialIntensity() = spatialAudio?.getIntensity() ?: 1.0f

    // ═══════════════════════════════════════════════════════════════
    // Content Analysis (for auto-presets)
    // ═══════════════════════════════════════════════════════════════

    fun submitForAnalysis(samples: FloatArray) = contentAnalyzer?.submitForAnalysis(samples)
    fun getContentType() = contentAnalyzer?.getContentType() ?: ContentAnalyzer.ContentType.UNKNOWN
    fun getRecommendedCompThreshold() = contentAnalyzer?.getRecommendedCompThreshold() ?: -20f
    fun getRecommendedBassBoost() = contentAnalyzer?.getRecommendedBassBoost() ?: 0f

    // ═══════════════════════════════════════════════════════════════
    // Statistics & Lifecycle
    // ═══════════════════════════════════════════════════════════════

    fun getStats() = AcceleratorStats(
        npuAvailable = npuAvailable.get(),
        gpuAvailable = gpuAvailable.get(),
        sourceSeparationAvailable = sourceSepAvailable.get(),
        spatialAudioAvailable = spatialAvailable.get(),
        sourceSeparatorBackend = sourceSeparatorBackend.get(),
        cpuProcessingMs = cpuProcessingTimeNs / 1_000_000f,
        npuProcessingMs = npuProcessingTimeNs / 1_000_000f,
        gpuProcessingMs = gpuProcessingTimeNs / 1_000_000f,
        totalOperations = totalOperations,
        processingMode = processingMode.get()
    )

    fun resetStats() {
        cpuProcessingTimeNs = 0; npuProcessingTimeNs = 0
        gpuProcessingTimeNs = 0; totalOperations = 0
    }

    fun release() {
        // PERFORMANCE: Proper shutdown with await to prevent resource leaks
        parallelExecutor?.let { executor ->
            executor.shutdown()
            try {
                if (!executor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                    executor.shutdownNow()
                }
            } catch (e: InterruptedException) {
                executor.shutdownNow()
                Thread.currentThread().interrupt()
            }
        }
        sourceSeparator?.release()
        spatialAudio?.release()
        contentAnalyzer?.shutdown()
        GpuCompute.release()

        npuAvailable.set(false)
        gpuAvailable.set(false)
        sourceSepAvailable.set(false)
        spatialAvailable.set(false)
        initialized.set(false)

        sourceSeparator = null
        spatialAudio = null
        contentAnalyzer = null
        parallelExecutor = null
    }

    fun isNpuAvailable() = npuAvailable.get()
    fun isGpuAvailable() = gpuAvailable.get()
    fun isSourceSeparationAvailable(): Boolean {
        val result = sourceSepAvailable.get()
        Timber.tag(TAG).d("isSourceSeparationAvailable() = $result")
        return result
    }

    fun isInitializationComplete() = initializationComplete.get()
    fun isSpatialAudioAvailable() = spatialAvailable.get()

    data class AcceleratorStats(
        val npuAvailable: Boolean,
        val gpuAvailable: Boolean,
        val sourceSeparationAvailable: Boolean,
        val spatialAudioAvailable: Boolean,
        val sourceSeparatorBackend: String,
        val cpuProcessingMs: Float,
        val npuProcessingMs: Float,
        val gpuProcessingMs: Float,
        val totalOperations: Long,
        val processingMode: ProcessingMode
    )
}
