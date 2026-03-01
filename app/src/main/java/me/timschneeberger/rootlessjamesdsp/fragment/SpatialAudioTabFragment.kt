package me.timschneeberger.rootlessjamesdsp.fragment

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.slider.Slider
import me.timschneeberger.rootlessjamesdsp.R
import me.timschneeberger.rootlessjamesdsp.audio.HardwareAccelerator
import me.timschneeberger.rootlessjamesdsp.utils.Constants
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.registerLocalReceiver
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.unregisterLocalReceiver
import me.timschneeberger.rootlessjamesdsp.utils.preferences.Preferences
import me.timschneeberger.rootlessjamesdsp.view.HeadTrackingVisualizerView
import org.koin.android.ext.android.inject
import timber.log.Timber
import kotlin.math.abs

/**
 * Spatial Audio Tab - 3D audio and head tracking controls
 */
class SpatialAudioTabFragment : Fragment() {

    // Preferences - needed to persist spatial audio state to service
    private val preferences: Preferences.App by inject()

    // Views
    private var switchSpatialAudio: MaterialSwitch? = null
    private var switchHeadTracking: MaterialSwitch? = null
    private var spatialStatus: TextView? = null
    private var headTrackingSource: TextView? = null
    private var headTrackingVisualizer: HeadTrackingVisualizerView? = null
    private var headPositionDisplay: LinearLayout? = null
    private var valueYaw: TextView? = null
    private var valuePitch: TextView? = null
    private var valueRoll: TextView? = null
    private var btnRecenter: MaterialButton? = null
    private var sliderRoomSize: Slider? = null
    private var sliderStereoWidth: Slider? = null
    private var valueRoomSize: TextView? = null
    private var valueStereoWidth: TextView? = null

    // Slider listeners (stored for cleanup in onDestroyView)
    private var roomSizeListener: Slider.OnChangeListener? = null
    private var stereoWidthListener: Slider.OnChangeListener? = null

    // Head tracking update handler
    private val headTrackingHandler = Handler(Looper.getMainLooper())
    private val headTrackingUpdateRunnable = object : Runnable {
        override fun run() {
            updateHeadTrackingVisualization()
            if (HardwareAccelerator.isHeadTrackingEnabled()) {
                headTrackingHandler.postDelayed(this, 33) // ~30fps
            }
        }
    }

    // Receiver for hardware accelerator ready broadcast
    // Fixes: Switch disabled on first launch because HardwareAccelerator initializes async
    private val hardwareReadyReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (!isAdded || view == null) return
            if (intent.action == Constants.ACTION_HARDWARE_ACCELERATOR_READY) {
                val spatialAvailable = intent.getBooleanExtra(Constants.EXTRA_SPATIAL_AUDIO_AVAILABLE, false)
                Timber.i("SpatialAudioTabFragment: Hardware ready broadcast received, spatial=$spatialAvailable")
                onHardwareAcceleratorReady(spatialAvailable)
            }
        }
    }

    // Callback to parent fragment
    var onSpatialStateChanged: ((Boolean) -> Unit)? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.tab_spatial_audio, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Find views
        switchSpatialAudio = view.findViewById(R.id.switch_spatial_audio)
        switchHeadTracking = view.findViewById(R.id.switch_head_tracking)
        spatialStatus = view.findViewById(R.id.spatial_status)
        headTrackingSource = view.findViewById(R.id.head_tracking_source)
        headTrackingVisualizer = view.findViewById(R.id.head_tracking_visualizer)
        headPositionDisplay = view.findViewById(R.id.head_position_display)
        valueYaw = view.findViewById(R.id.value_yaw)
        valuePitch = view.findViewById(R.id.value_pitch)
        valueRoll = view.findViewById(R.id.value_roll)
        btnRecenter = view.findViewById(R.id.btn_recenter)
        sliderRoomSize = view.findViewById(R.id.slider_room_size)
        sliderStereoWidth = view.findViewById(R.id.slider_stereo_width)
        valueRoomSize = view.findViewById(R.id.value_room_size)
        valueStereoWidth = view.findViewById(R.id.value_stereo_width)

        initializeSpatialAudio()
        setupListeners()

        Timber.d("SpatialAudioTabFragment initialized")
    }

    private fun initializeSpatialAudio() {
        val ctx = context ?: return

        try {
            // Check if hardware accelerator initialization is still in progress
            if (!HardwareAccelerator.isInitializationComplete()) {
                // Initialization still in progress, retry after a delay
                view?.postDelayed({ initializeSpatialAudio() }, 500)
                return
            }

            // Use HardwareAccelerator's shared instance instead of creating a new one
            if (HardwareAccelerator.isSpatialAudioAvailable()) {
                spatialStatus?.text = "Ready"
                spatialStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_success))
                Timber.i("Spatial Audio available via HardwareAccelerator")
            } else {
                spatialStatus?.text = "Unavailable"
                spatialStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_error))
                switchSpatialAudio?.isEnabled = false
                switchHeadTracking?.isEnabled = false
                Timber.w("Spatial Audio not available")
            }

            // Load saved intensity value from preferences
            // Coerce to valid slider range (0-200) to prevent IllegalStateException
            val savedIntensity = preferences.get<Float>(R.string.key_spatial_intensity, 100f).coerceIn(0f, 200f)
            sliderStereoWidth?.value = savedIntensity
            valueStereoWidth?.text = "${savedIntensity.toInt()}%"

            // Initially disable head tracking switch until spatial audio is enabled
            switchHeadTracking?.isEnabled = false
        } catch (e: Exception) {
            Timber.e(e, "Failed to check Spatial Audio status")
            spatialStatus?.text = "Error"
            spatialStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_error))
            switchSpatialAudio?.isEnabled = false
            switchHeadTracking?.isEnabled = false
        }

        // Restore switch states from preferences (must be after setupListeners is called)
        // We post this to ensure listeners are set up first
        view?.post {
            // Safety check - fragment may have been detached before post runs
            if (!isAdded || view == null) return@post

            val spatialWasEnabled = preferences.get<Boolean>(R.string.key_spatial_audio_enable, false)
            switchSpatialAudio?.isChecked = spatialWasEnabled

            // Only restore head tracking if spatial audio is enabled
            if (spatialWasEnabled) {
                val headTrackingWasEnabled = preferences.get<Boolean>(R.string.key_spatial_headtracking_enable, false)
                switchHeadTracking?.isChecked = headTrackingWasEnabled
            }
        }
    }

    private fun setupListeners() {
        val ctx = context ?: return

        // Main spatial audio switch
        switchSpatialAudio?.setOnCheckedChangeListener { _, isChecked ->
            // CRITICAL: Save to SharedPreferences so the service can read it
            preferences.set(R.string.key_spatial_audio_enable, isChecked)

            if (isChecked) {
                spatialStatus?.text = "Active"
                spatialStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_success))
                switchHeadTracking?.isEnabled = true
            } else {
                spatialStatus?.text = "Ready"
                spatialStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_text_tertiary))
                if (switchHeadTracking?.isChecked == true) {
                    switchHeadTracking?.isChecked = false
                }
                switchHeadTracking?.isEnabled = false
            }
            onSpatialStateChanged?.invoke(isChecked)
            Timber.i("Spatial Audio: ${if (isChecked) "enabled" else "disabled"}")
        }

        // Head tracking switch
        switchHeadTracking?.setOnCheckedChangeListener { _, isChecked ->
            // CRITICAL: Save to SharedPreferences so the service can read it
            preferences.set(R.string.key_spatial_headtracking_enable, isChecked)

            if (isChecked) {
                HardwareAccelerator.enableHeadTracking()
                // Head tracking source is determined by HardwareAccelerator
                headTrackingSource?.text = "Phone Gyroscope"
                headTrackingSource?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_warning))

                headTrackingVisualizer?.isVisible = true
                headPositionDisplay?.isVisible = true
                btnRecenter?.isVisible = true
                headTrackingHandler.post(headTrackingUpdateRunnable)
            } else {
                HardwareAccelerator.disableHeadTracking()
                headTrackingHandler.removeCallbacks(headTrackingUpdateRunnable)
                headTrackingVisualizer?.isVisible = false
                headPositionDisplay?.isVisible = false
                btnRecenter?.isVisible = false
                headTrackingVisualizer?.reset()
            }
            Timber.i("Head Tracking: ${if (isChecked) "enabled" else "disabled"}")
        }

        // Recenter button
        btnRecenter?.setOnClickListener {
            HardwareAccelerator.recenterHeadTracking()
            it.animate()
                .scaleX(0.9f)
                .scaleY(0.9f)
                .setDuration(100)
                .withEndAction {
                    it.animate()
                        .scaleX(1f)
                        .scaleY(1f)
                        .setDuration(100)
                        .start()
                }
                .start()
            Timber.i("Head tracking recentered")
        }

        // Room size slider - Note: Currently cosmetic, no backing native implementation
        roomSizeListener = Slider.OnChangeListener { _, value, fromUser ->
            if (!fromUser) return@OnChangeListener
            valueRoomSize?.text = "${value.toInt()}%"
            // TODO: Add native room size support when HRTF reverb tail is implemented
        }
        roomSizeListener?.let { sliderRoomSize?.addOnChangeListener(it) }

        // Stereo width / intensity slider - Controls spatial effect intensity
        stereoWidthListener = Slider.OnChangeListener { _, value, fromUser ->
            if (!fromUser) return@OnChangeListener
            val intensity = value / 100f  // Convert 0-200 to 0.0-2.0
            valueStereoWidth?.text = "${value.toInt()}%"
            // Apply intensity to spatial audio processor
            HardwareAccelerator.setSpatialIntensity(intensity)
            // Save to preferences so service can restore on restart
            preferences.set(R.string.key_spatial_intensity, value)
        }
        stereoWidthListener?.let { sliderStereoWidth?.addOnChangeListener(it) }
    }

    private fun updateHeadTrackingVisualization() {
        if (!HardwareAccelerator.isHeadTrackingEnabled()) return

        val yaw = HardwareAccelerator.getHeadAzimuth()
        val pitch = HardwareAccelerator.getHeadElevation()
        val roll = 0f  // Roll not tracked via HardwareAccelerator currently

        headTrackingVisualizer?.updateHeadPose(yaw, pitch, roll)

        valueYaw?.text = "${yaw.toInt()}°"
        valuePitch?.text = "${pitch.toInt()}°"
        valueRoll?.text = "${roll.toInt()}°"

        context?.let { ctx ->
            valueYaw?.setTextColor(ContextCompat.getColor(ctx,
                if (abs(yaw) > 45f) R.color.ultra_warning else R.color.ultra_success))
            valuePitch?.setTextColor(ContextCompat.getColor(ctx,
                if (abs(pitch) > 30f) R.color.ultra_warning else R.color.ultra_info))
            valueRoll?.setTextColor(ContextCompat.getColor(ctx,
                if (abs(roll) > 15f) R.color.ultra_warning else R.color.ultra_secondary))
        }
    }

    /**
     * Called when HardwareAccelerator finishes async initialization
     * Re-checks spatial audio availability and enables the switch if available
     */
    private fun onHardwareAcceleratorReady(spatialAvailable: Boolean) {
        val ctx = context ?: return

        if (spatialAvailable) {
            spatialStatus?.text = "Ready"
            spatialStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_success))
            switchSpatialAudio?.isEnabled = true
            Timber.i("Spatial Audio now available - switch enabled")

            // Restore switch state from preferences if it was previously enabled
            val wasEnabled = preferences.get<Boolean>(R.string.key_spatial_audio_enable, false)
            if (wasEnabled && switchSpatialAudio?.isChecked != true) {
                switchSpatialAudio?.isChecked = true
            }
        } else {
            spatialStatus?.text = "Unavailable"
            spatialStatus?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_error))
            switchSpatialAudio?.isEnabled = false
            switchHeadTracking?.isEnabled = false
        }
    }

    fun isEnabled(): Boolean = switchSpatialAudio?.isChecked == true

    fun setEnabled(enabled: Boolean) {
        switchSpatialAudio?.isChecked = enabled
    }

    override fun onResume() {
        super.onResume()
        // Register for hardware accelerator ready broadcast
        // This allows the switch to be enabled if HardwareAccelerator finishes init after fragment is visible
        context?.registerLocalReceiver(hardwareReadyReceiver, IntentFilter(Constants.ACTION_HARDWARE_ACCELERATOR_READY))

        // Also re-check availability in case init finished while we were paused
        if (HardwareAccelerator.isSpatialAudioAvailable() && switchSpatialAudio?.isEnabled == false) {
            onHardwareAcceleratorReady(true)
        }
    }

    override fun onPause() {
        // Unregister hardware ready receiver
        context?.unregisterLocalReceiver(hardwareReadyReceiver)

        // Stop head tracking updates first
        headTrackingHandler.removeCallbacks(headTrackingUpdateRunnable)

        // Reset visualizer to stop animations before surface transitions
        headTrackingVisualizer?.reset()

        super.onPause()
    }

    override fun onDestroyView() {
        headTrackingHandler.removeCallbacks(headTrackingUpdateRunnable)

        // MEMORY LEAK FIX: Cancel any running animations
        btnRecenter?.animate()?.cancel()

        // Remove slider listeners
        roomSizeListener?.let { sliderRoomSize?.removeOnChangeListener(it) }
        stereoWidthListener?.let { sliderStereoWidth?.removeOnChangeListener(it) }
        roomSizeListener = null
        stereoWidthListener = null

        switchSpatialAudio?.setOnCheckedChangeListener(null)
        switchHeadTracking?.setOnCheckedChangeListener(null)
        btnRecenter?.setOnClickListener(null)

        // Note: Don't release HardwareAccelerator here - it's shared and managed by the service

        switchSpatialAudio = null
        switchHeadTracking = null
        spatialStatus = null
        headTrackingSource = null
        headTrackingVisualizer = null
        headPositionDisplay = null
        valueYaw = null
        valuePitch = null
        valueRoll = null
        btnRecenter = null
        sliderRoomSize = null
        sliderStereoWidth = null
        valueRoomSize = null
        valueStereoWidth = null

        super.onDestroyView()
    }

    companion object {
        fun newInstance() = SpatialAudioTabFragment()
    }
}
