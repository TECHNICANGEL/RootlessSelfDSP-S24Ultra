package me.timschneeberger.rootlessjamesdsp.fragment

import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.animation.DecelerateInterpolator
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.core.widget.NestedScrollView
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.slider.Slider
import me.timschneeberger.rootlessjamesdsp.R
import me.timschneeberger.rootlessjamesdsp.audio.HardwareAccelerator
import me.timschneeberger.rootlessjamesdsp.audio.SourceSeparator
import me.timschneeberger.rootlessjamesdsp.utils.Constants
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.registerLocalReceiver
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.unregisterLocalReceiver
import me.timschneeberger.rootlessjamesdsp.utils.preferences.Preferences
import me.timschneeberger.rootlessjamesdsp.view.NpuStatsView
import me.timschneeberger.rootlessjamesdsp.view.SeparationStateView
import me.timschneeberger.rootlessjamesdsp.view.StemLevelView
import org.koin.android.ext.android.inject
import timber.log.Timber

/**
 * Source Separation Tab - AI-powered stem separation controls
 */
class SourceSeparationTabFragment : Fragment() {

    // Preferences - needed to persist source separation state to service
    private val preferences: Preferences.App by inject()

    // Receiver for stem level visualization data
    private val stemLevelReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (!isAdded || view == null) return
            if (intent.action == Constants.ACTION_STEM_LEVELS) {
                val vocals = intent.getFloatExtra(Constants.EXTRA_STEM_VOCALS_LEVEL, 0f)
                val drums = intent.getFloatExtra(Constants.EXTRA_STEM_DRUMS_LEVEL, 0f)
                val bass = intent.getFloatExtra(Constants.EXTRA_STEM_BASS_LEVEL, 0f)
                val other = intent.getFloatExtra(Constants.EXTRA_STEM_OTHER_LEVEL, 0f)
                stemLevelView?.updateLevels(vocals, drums, bass, other)
            }
        }
    }

    // Receiver for hardware accelerator ready broadcast
    // Fixes: Switch disabled on first launch because HardwareAccelerator initializes async
    private val hardwareReadyReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (!isAdded || view == null) return
            if (intent.action == Constants.ACTION_HARDWARE_ACCELERATOR_READY) {
                val separationAvailable = intent.getBooleanExtra(Constants.EXTRA_SOURCE_SEPARATION_AVAILABLE, false)
                Timber.i("SourceSeparationTabFragment: Hardware ready broadcast received, separation=$separationAvailable")
                onHardwareAcceleratorReady(separationAvailable)
            }
        }
    }

    // Receiver for separation state updates (buffering, processing, playing)
    private val separationStateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (!isAdded || view == null) return
            if (intent.action == Constants.ACTION_SEPARATION_STATE) {
                val stateOrdinal = intent.getIntExtra(Constants.EXTRA_SEPARATION_STATE, 0)
                val progress = intent.getFloatExtra(Constants.EXTRA_SEPARATION_PROGRESS, 0f)
                val message = intent.getStringExtra(Constants.EXTRA_SEPARATION_MESSAGE) ?: ""
                val npuUsage = intent.getFloatExtra(Constants.EXTRA_NPU_USAGE, 0f)
                val state = SourceSeparator.SeparationState.entries.getOrElse(stateOrdinal) {
                    SourceSeparator.SeparationState.IDLE
                }
                separationStateView?.updateState(state, progress, message, npuUsage)
            }
        }
    }

    // Receiver for pipeline timing stats
    private val pipelineStatsReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (!isAdded || view == null) return
            if (intent.action == Constants.ACTION_PIPELINE_STATS) {
                val bufferPct = intent.getIntExtra(Constants.EXTRA_BUFFER_PCT, 0)
                val stftMs = intent.getIntExtra(Constants.EXTRA_STFT_MS, 0)
                val npuMs = intent.getIntExtra(Constants.EXTRA_NPU_MS, 0)
                val istftMs = intent.getIntExtra(Constants.EXTRA_ISTFT_MS, 0)
                val queueSize = intent.getIntExtra(Constants.EXTRA_QUEUE_SIZE, 0)
                val stateOrdinal = intent.getIntExtra(Constants.EXTRA_SEPARATION_STATE, 0)
                val progress = intent.getFloatExtra(Constants.EXTRA_SEPARATION_PROGRESS, 0f)

                // Get state from the broadcast
                val state = SourceSeparator.SeparationState.entries.getOrElse(stateOrdinal) {
                    SourceSeparator.SeparationState.IDLE
                }

                // Update separation state view with current state and progress
                separationStateView?.updateState(state, progress, "Q:$queueSize", 0f)
                separationStateView?.updatePipelineStats(bufferPct, stftMs, npuMs, istftMs, queueSize)

                // Update NPU stats view with timing data
                val totalMs = stftMs + npuMs + istftMs
                val chunkMs = 2000f  // 2 second chunks
                val rtFactor = if (totalMs > 0) chunkMs / totalMs else 0f
                val npuUsage = if (chunkMs > 0) (totalMs / chunkMs) * 100f else 0f
                npuStatsView?.updateStats(
                    npuUsage = npuUsage,
                    rtFactor = rtFactor,
                    stftTime = stftMs,
                    npuTime = npuMs,
                    istftTime = istftMs,
                    queue = queueSize,
                    currentState = state.name,
                    bufferPct = bufferPct
                )
            }
        }
    }

    // Receiver for individual model timing stats
    private val modelStatsReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (!isAdded || view == null) return
            if (intent.action == Constants.ACTION_MODEL_STATS) {
                val vocalsMs = intent.getIntExtra(Constants.EXTRA_VOCALS_MS, 0)
                val drumsMs = intent.getIntExtra(Constants.EXTRA_DRUMS_MS, 0)
                val bassMs = intent.getIntExtra(Constants.EXTRA_BASS_MS, 0)
                val otherMs = intent.getIntExtra(Constants.EXTRA_OTHER_MS, 0)
                val totalMs = intent.getIntExtra(Constants.EXTRA_TOTAL_MS, 0)
                val usingNpu = intent.getBooleanExtra(Constants.EXTRA_USING_NPU, false)
                val backendName = intent.getStringExtra(Constants.EXTRA_BACKEND_NAME) ?: "CPU"

                npuStatsView?.updateModelStats(vocalsMs, drumsMs, bassMs, otherMs, totalMs, usingNpu, backendName)
            }
        }
    }

    // Views
    private var scrollView: NestedScrollView? = null
    private var switchSourceSeparation: MaterialSwitch? = null
    private var separationStatus: TextView? = null
    private var labelStatus: TextView? = null
    private var labelLiveOutput: TextView? = null
    private var separationStateView: SeparationStateView? = null
    private var npuStatsView: NpuStatsView? = null
    private var stemLevelView: StemLevelView? = null
    private var sliderVocals: Slider? = null
    private var sliderDrums: Slider? = null
    private var sliderBass: Slider? = null
    private var sliderOther: Slider? = null
    private var valueVocals: TextView? = null
    private var valueDrums: TextView? = null
    private var valueBass: TextView? = null
    private var valueOther: TextView? = null
    private var btnPresetNormal: MaterialButton? = null
    private var btnPresetKaraoke: MaterialButton? = null
    private var btnPresetVocals: MaterialButton? = null
    private var btnPresetBassBoost: MaterialButton? = null
    private var btnPresetDrumsBoost: MaterialButton? = null

    // Slider listener
    private var sliderListener: Slider.OnChangeListener? = null

    // Track backend name for UI display
    private var backendName: String = "Spectral"

    // Callback to parent fragment
    var onSeparationStateChanged: ((Boolean) -> Unit)? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.tab_source_separation, container, false)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Find views
        scrollView = view.findViewById(R.id.scroll_view)
        switchSourceSeparation = view.findViewById(R.id.switch_source_separation)
        separationStatus = view.findViewById(R.id.separation_status)
        labelStatus = view.findViewById(R.id.label_status)
        labelLiveOutput = view.findViewById(R.id.label_live_output)
        separationStateView = view.findViewById(R.id.separation_state_view)
        npuStatsView = view.findViewById(R.id.npu_stats_view)
        stemLevelView = view.findViewById(R.id.stem_level_view)
        sliderVocals = view.findViewById(R.id.slider_vocals)
        sliderDrums = view.findViewById(R.id.slider_drums)
        sliderBass = view.findViewById(R.id.slider_bass)
        sliderOther = view.findViewById(R.id.slider_other)
        valueVocals = view.findViewById(R.id.value_vocals)
        valueDrums = view.findViewById(R.id.value_drums)
        valueBass = view.findViewById(R.id.value_bass)
        valueOther = view.findViewById(R.id.value_other)
        btnPresetNormal = view.findViewById(R.id.btn_preset_normal)
        btnPresetKaraoke = view.findViewById(R.id.btn_preset_karaoke)
        btnPresetVocals = view.findViewById(R.id.btn_preset_vocals)
        btnPresetBassBoost = view.findViewById(R.id.btn_preset_bass_boost)
        btnPresetDrumsBoost = view.findViewById(R.id.btn_preset_drums_boost)

        // Fix slider touch handling - prevent scroll view from intercepting horizontal drags
        setupSliderTouchHandling()

        initializeSourceSeparator()
        setupListeners()

        Timber.d("SourceSeparationTabFragment initialized")
    }

    /**
     * Prevents the parent NestedScrollView from intercepting touch events when
     * the user is dragging a slider horizontally.
     */
    @SuppressLint("ClickableViewAccessibility")
    private fun setupSliderTouchHandling() {
        val sliders = listOf(sliderVocals, sliderDrums, sliderBass, sliderOther)

        sliders.forEach { slider ->
            slider?.setOnTouchListener { v, event ->
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        // Tell parent to not intercept touch events
                        scrollView?.requestDisallowInterceptTouchEvent(true)
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        // Allow parent to intercept again
                        scrollView?.requestDisallowInterceptTouchEvent(false)
                    }
                }
                // Return false to let the slider handle the event
                false
            }
        }
    }

    private fun initializeSourceSeparator() {
        val ctx = context ?: return

        try {
            // Check if hardware accelerator initialization is still in progress
            if (!HardwareAccelerator.isInitializationComplete()) {
                // Initialization still in progress, retry after a delay
                view?.postDelayed({ initializeSourceSeparator() }, 500)
                return
            }

            // Use HardwareAccelerator's shared instance instead of creating a new one
            // This avoids double initialization and ensures UI controls the actual processor
            if (HardwareAccelerator.isSourceSeparationAvailable()) {
                // Get actual backend name from HardwareAccelerator
                val stats = HardwareAccelerator.getStats()
                backendName = stats.sourceSeparatorBackend

                // Color code based on backend type
                val statusColor = when {
                    backendName.contains("NPU", ignoreCase = true) ||
                    backendName.contains("Hexagon", ignoreCase = true) -> R.color.ultra_success
                    backendName.contains("CPU", ignoreCase = true) ||
                    backendName.contains("Spectral", ignoreCase = true) -> R.color.ultra_warning
                    backendName.contains("GPU", ignoreCase = true) -> R.color.ultra_info
                    else -> R.color.ultra_text_tertiary
                }
                separationStatus?.text = "$backendName • Ready"
                separationStatus?.setTextColor(ContextCompat.getColor(ctx, statusColor))
                Timber.i("Source Separator available: $backendName")
            } else {
                backendName = "Unavailable"
                separationStatus?.text = "Unavailable"
                separationStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_error))
                switchSourceSeparation?.isEnabled = false
                Timber.w("Source Separator not available")
            }

            // Load saved stem levels from preferences
            val vocals = preferences.get<Float>(R.string.key_stem_vocals_level, 100f)
            val drums = preferences.get<Float>(R.string.key_stem_drums_level, 100f)
            val bass = preferences.get<Float>(R.string.key_stem_bass_level, 100f)
            val other = preferences.get<Float>(R.string.key_stem_other_level, 100f)
            updateStemSliders(vocals, drums, bass, other)

            // Restore switch state from preferences (must be after setupListeners is called)
            // We post this to ensure listeners are set up first
            view?.post {
                // Safety check - fragment may have been detached before post runs
                if (!isAdded || view == null) return@post

                val wasEnabled = preferences.get<Boolean>(R.string.key_source_separation_enable, false)
                switchSourceSeparation?.isChecked = wasEnabled
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to check Source Separator status")
            backendName = "Error"
            separationStatus?.text = "Error"
            separationStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_error))
            switchSourceSeparation?.isEnabled = false
        }
    }

    private fun setupListeners() {
        val ctx = context ?: return

        // Main switch
        switchSourceSeparation?.setOnCheckedChangeListener { _, isChecked ->
            // CRITICAL: Save to SharedPreferences so the service can read it
            // The service listens for preference changes and updates sourceSeparationEnabled
            preferences.set(R.string.key_source_separation_enable, isChecked)

            if (isChecked) {
                val stats = HardwareAccelerator.getStats()
                backendName = stats.sourceSeparatorBackend
                val statusColor = when {
                    backendName.contains("NPU", ignoreCase = true) ||
                    backendName.contains("Hexagon", ignoreCase = true) -> R.color.ultra_success
                    backendName.contains("CPU", ignoreCase = true) ||
                    backendName.contains("Spectral", ignoreCase = true) -> R.color.ultra_warning
                    backendName.contains("GPU", ignoreCase = true) -> R.color.ultra_info
                    else -> R.color.ultra_text_tertiary
                }
                separationStatus?.text = "$backendName • Active"
                separationStatus?.setTextColor(ContextCompat.getColor(ctx, statusColor))

                // Show processing section
                labelStatus?.isVisible = true
                separationStateView?.isVisible = true
                npuStatsView?.isVisible = true

                // Show live output section
                labelLiveOutput?.isVisible = true
                stemLevelView?.isVisible = true
            } else {
                separationStatus?.text = "$backendName • Ready"
                separationStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_text_tertiary))

                // Hide processing section
                labelStatus?.isVisible = false
                separationStateView?.isVisible = false
                npuStatsView?.isVisible = false
                npuStatsView?.reset()

                // Hide live output section
                labelLiveOutput?.isVisible = false
                stemLevelView?.isVisible = false
                stemLevelView?.reset()
            }
            onSeparationStateChanged?.invoke(isChecked)
            Timber.i("Source Separation ($backendName): ${if (isChecked) "enabled" else "disabled"}")
        }

        // Stem sliders - save to preferences so service can use them
        // HardwareAccelerator will apply these via preference change listener in service
        sliderListener = Slider.OnChangeListener { slider, value, fromUser ->
            if (!fromUser) return@OnChangeListener
            val level = value / 100f
            val levelDisplay = value.toInt()
            when (slider.id) {
                R.id.slider_vocals -> {
                    HardwareAccelerator.setVocalsLevel(level)
                    preferences.set(R.string.key_stem_vocals_level, value)
                    valueVocals?.text = "$levelDisplay%"
                }
                R.id.slider_drums -> {
                    HardwareAccelerator.setDrumsLevel(level)
                    preferences.set(R.string.key_stem_drums_level, value)
                    valueDrums?.text = "$levelDisplay%"
                }
                R.id.slider_bass -> {
                    HardwareAccelerator.setBassLevel(level)
                    preferences.set(R.string.key_stem_bass_level, value)
                    valueBass?.text = "$levelDisplay%"
                }
                R.id.slider_other -> {
                    HardwareAccelerator.setOtherLevel(level)
                    preferences.set(R.string.key_stem_other_level, value)
                    valueOther?.text = "$levelDisplay%"
                }
            }
        }

        sliderListener?.let { listener ->
            sliderVocals?.addOnChangeListener(listener)
            sliderDrums?.addOnChangeListener(listener)
            sliderBass?.addOnChangeListener(listener)
            sliderOther?.addOnChangeListener(listener)
        }

        // Preset buttons - use HardwareAccelerator presets
        btnPresetNormal?.setOnClickListener {
            HardwareAccelerator.resetStemLevels()
            updateStemSliders(100f, 100f, 100f, 100f)
            saveStemLevelsToPrefs(100f, 100f, 100f, 100f)
            highlightPresetButton(btnPresetNormal)
        }

        btnPresetKaraoke?.setOnClickListener {
            HardwareAccelerator.setKaraokeMode()
            updateStemSliders(0f, 100f, 100f, 100f)
            saveStemLevelsToPrefs(0f, 100f, 100f, 100f)
            highlightPresetButton(btnPresetKaraoke)
        }

        btnPresetVocals?.setOnClickListener {
            HardwareAccelerator.setVocalsOnlyMode()
            updateStemSliders(100f, 0f, 0f, 0f)
            saveStemLevelsToPrefs(100f, 0f, 0f, 0f)
            highlightPresetButton(btnPresetVocals)
        }

        btnPresetBassBoost?.setOnClickListener {
            HardwareAccelerator.setVocalsLevel(0.9f)
            HardwareAccelerator.setDrumsLevel(1.0f)
            HardwareAccelerator.setBassLevel(1.5f)
            HardwareAccelerator.setOtherLevel(0.9f)
            updateStemSliders(90f, 100f, 150f, 90f)
            saveStemLevelsToPrefs(90f, 100f, 150f, 90f)
            highlightPresetButton(btnPresetBassBoost)
        }

        btnPresetDrumsBoost?.setOnClickListener {
            HardwareAccelerator.setVocalsLevel(0.8f)
            HardwareAccelerator.setDrumsLevel(1.5f)
            HardwareAccelerator.setBassLevel(1.2f)
            HardwareAccelerator.setOtherLevel(0.8f)
            updateStemSliders(80f, 150f, 120f, 80f)
            saveStemLevelsToPrefs(80f, 150f, 120f, 80f)
            highlightPresetButton(btnPresetDrumsBoost)
        }

        // Highlight the preset button that matches the saved stem levels
        highlightMatchingPreset()
    }

    /**
     * Determines which preset button should be highlighted based on current stem values
     */
    private fun highlightMatchingPreset() {
        val vocals = sliderVocals?.value ?: 100f
        val drums = sliderDrums?.value ?: 100f
        val bass = sliderBass?.value ?: 100f
        val other = sliderOther?.value ?: 100f

        val matchingButton = when {
            // Normal: 100, 100, 100, 100
            vocals == 100f && drums == 100f && bass == 100f && other == 100f -> btnPresetNormal
            // Karaoke: 0, 100, 100, 100
            vocals == 0f && drums == 100f && bass == 100f && other == 100f -> btnPresetKaraoke
            // Vocals Only: 100, 0, 0, 0
            vocals == 100f && drums == 0f && bass == 0f && other == 0f -> btnPresetVocals
            // Bass Boost: 90, 100, 150, 90
            vocals == 90f && drums == 100f && bass == 150f && other == 90f -> btnPresetBassBoost
            // Drums Boost: 80, 150, 120, 80
            vocals == 80f && drums == 150f && bass == 120f && other == 80f -> btnPresetDrumsBoost
            // Custom values - don't highlight any preset
            else -> null
        }

        highlightPresetButton(matchingButton)
    }

    private fun saveStemLevelsToPrefs(vocals: Float, drums: Float, bass: Float, other: Float) {
        preferences.set(R.string.key_stem_vocals_level, vocals)
        preferences.set(R.string.key_stem_drums_level, drums)
        preferences.set(R.string.key_stem_bass_level, bass)
        preferences.set(R.string.key_stem_other_level, other)
    }

    private fun updateStemSliders(vocals: Float, drums: Float, bass: Float, other: Float) {
        // Coerce values to valid slider range (0-200) to prevent IllegalStateException
        sliderVocals?.value = vocals.coerceIn(0f, 200f)
        sliderDrums?.value = drums.coerceIn(0f, 200f)
        sliderBass?.value = bass.coerceIn(0f, 200f)
        sliderOther?.value = other.coerceIn(0f, 200f)
        valueVocals?.text = "${vocals.toInt()}%"
        valueDrums?.text = "${drums.toInt()}%"
        valueBass?.text = "${bass.toInt()}%"
        valueOther?.text = "${other.toInt()}%"
    }

    private fun highlightPresetButton(activeButton: MaterialButton?) {
        val ctx = context ?: return
        val buttons = listOf(btnPresetNormal, btnPresetKaraoke, btnPresetVocals, btnPresetBassBoost, btnPresetDrumsBoost)

        buttons.forEach { button ->
            if (button == activeButton) {
                button?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_primary))
                button?.strokeColor = ContextCompat.getColorStateList(ctx, R.color.ultra_primary)
                button?.strokeWidth = 2

                button?.animate()
                    ?.scaleX(1.05f)
                    ?.scaleY(1.05f)
                    ?.setDuration(100)
                    ?.setInterpolator(DecelerateInterpolator())
                    ?.withEndAction {
                        // Use null-safe call in case button becomes null during animation
                        button?.animate()
                            ?.scaleX(1f)
                            ?.scaleY(1f)
                            ?.setDuration(100)
                            ?.start()
                    }
                    ?.start()
            } else {
                button?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_text_tertiary))
                button?.strokeColor = ContextCompat.getColorStateList(ctx, R.color.ultra_glass_border)
                button?.strokeWidth = 1
            }
        }
    }

    fun isEnabled(): Boolean = switchSourceSeparation?.isChecked == true

    fun setEnabled(enabled: Boolean) {
        switchSourceSeparation?.isChecked = enabled
    }

    /**
     * Called when HardwareAccelerator finishes async initialization
     * Re-checks source separation availability and enables the switch if available
     */
    private fun onHardwareAcceleratorReady(separationAvailable: Boolean) {
        val ctx = context ?: return

        if (separationAvailable) {
            // Get actual backend name from HardwareAccelerator
            val stats = HardwareAccelerator.getStats()
            backendName = stats.sourceSeparatorBackend

            // Color code based on backend type
            val statusColor = when {
                backendName.contains("NPU", ignoreCase = true) ||
                backendName.contains("Hexagon", ignoreCase = true) -> R.color.ultra_success
                backendName.contains("CPU", ignoreCase = true) ||
                backendName.contains("Spectral", ignoreCase = true) -> R.color.ultra_warning
                backendName.contains("GPU", ignoreCase = true) -> R.color.ultra_info
                else -> R.color.ultra_text_tertiary
            }
            separationStatus?.text = "$backendName • Ready"
            separationStatus?.setTextColor(ContextCompat.getColor(ctx, statusColor))
            switchSourceSeparation?.isEnabled = true
            Timber.i("Source Separation now available ($backendName) - switch enabled")

            // Restore switch state from preferences if it was previously enabled
            val wasEnabled = preferences.get<Boolean>(R.string.key_source_separation_enable, false)
            if (wasEnabled && switchSourceSeparation?.isChecked != true) {
                switchSourceSeparation?.isChecked = true
            }
        } else {
            backendName = "Unavailable"
            separationStatus?.text = "Unavailable"
            separationStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_error))
            switchSourceSeparation?.isEnabled = false
        }
    }

    override fun onResume() {
        super.onResume()
        // Register for stem level visualization updates
        context?.registerLocalReceiver(stemLevelReceiver, IntentFilter(Constants.ACTION_STEM_LEVELS))
        // Register for hardware accelerator ready broadcast
        context?.registerLocalReceiver(hardwareReadyReceiver, IntentFilter(Constants.ACTION_HARDWARE_ACCELERATOR_READY))
        // Register for separation state updates
        context?.registerLocalReceiver(separationStateReceiver, IntentFilter(Constants.ACTION_SEPARATION_STATE))
        // Register for pipeline timing stats
        context?.registerLocalReceiver(pipelineStatsReceiver, IntentFilter(Constants.ACTION_PIPELINE_STATS))
        // Register for individual model timing stats
        context?.registerLocalReceiver(modelStatsReceiver, IntentFilter(Constants.ACTION_MODEL_STATS))

        // Also re-check availability in case init finished while we were paused
        if (HardwareAccelerator.isSourceSeparationAvailable() && switchSourceSeparation?.isEnabled == false) {
            onHardwareAcceleratorReady(true)
        }
    }

    override fun onPause() {
        // Unregister receivers
        context?.unregisterLocalReceiver(stemLevelReceiver)
        context?.unregisterLocalReceiver(hardwareReadyReceiver)
        context?.unregisterLocalReceiver(separationStateReceiver)
        context?.unregisterLocalReceiver(pipelineStatsReceiver)
        context?.unregisterLocalReceiver(modelStatsReceiver)
        // Reset stem level visualizer to stop animations before surface transitions
        stemLevelView?.reset()
        super.onPause()
    }

    override fun onDestroyView() {
        sliderListener?.let { listener ->
            sliderVocals?.removeOnChangeListener(listener)
            sliderDrums?.removeOnChangeListener(listener)
            sliderBass?.removeOnChangeListener(listener)
            sliderOther?.removeOnChangeListener(listener)
        }
        sliderListener = null

        // Clear touch listeners
        sliderVocals?.setOnTouchListener(null)
        sliderDrums?.setOnTouchListener(null)
        sliderBass?.setOnTouchListener(null)
        sliderOther?.setOnTouchListener(null)

        switchSourceSeparation?.setOnCheckedChangeListener(null)
        btnPresetNormal?.setOnClickListener(null)
        btnPresetKaraoke?.setOnClickListener(null)
        btnPresetVocals?.setOnClickListener(null)
        btnPresetBassBoost?.setOnClickListener(null)
        btnPresetDrumsBoost?.setOnClickListener(null)

        // Note: Don't release HardwareAccelerator here - it's shared and managed by the service

        scrollView = null
        switchSourceSeparation = null
        separationStatus = null
        labelStatus = null
        labelLiveOutput = null
        separationStateView = null
        npuStatsView = null
        stemLevelView = null
        sliderVocals = null
        sliderDrums = null
        sliderBass = null
        sliderOther = null
        valueVocals = null
        valueDrums = null
        valueBass = null
        valueOther = null
        btnPresetNormal = null
        btnPresetKaraoke = null
        btnPresetVocals = null
        btnPresetBassBoost = null
        btnPresetDrumsBoost = null

        super.onDestroyView()
    }

    companion object {
        fun newInstance() = SourceSeparationTabFragment()
    }
}
