package me.timschneeberger.rootlessjamesdsp.fragment

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import me.timschneeberger.rootlessjamesdsp.R
import me.timschneeberger.rootlessjamesdsp.utils.Constants
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.registerLocalReceiver
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.unregisterLocalReceiver
import me.timschneeberger.rootlessjamesdsp.view.StatsMonitorView
import me.timschneeberger.rootlessjamesdsp.view.VUMeterView
import me.timschneeberger.rootlessjamesdsp.view.VulkanSpectrumView
import timber.log.Timber

/**
 * Dashboard Tab - Real-time audio visualization
 * Contains: Vulkan RT Spectrum Analyzer, VU Meters, Quick Stats
 *
 * S24 ULTRA ONLY - Vulkan with Hardware Ray Tracing
 * No fallback. Adreno 750 or nothing.
 */
class DashboardTabFragment : Fragment() {

    // Views - Vulkan RT only, no fallbacks
    private var vulkanSpectrumAnalyzer: VulkanSpectrumView? = null
    private var chipVulkanRt: Chip? = null
    private var vuMeter: VUMeterView? = null
    private var statsMonitor: StatsMonitorView? = null
    private var quickLatency: TextView? = null
    private var quickLatencyStatus: TextView? = null
    private var quickSampleRate: TextView? = null
    private var quickSampleStatus: TextView? = null
    private var quickBuffer: TextView? = null
    private var quickBufferStatus: TextView? = null

    // Visual effects controls
    private var chipGroupTheme: ChipGroup? = null
    private var chipGroupStereo: ChipGroup? = null

    // State tracking for edge cases
    private var isReceiverRegistered = false
    private var pendingRunnables = mutableListOf<Runnable>()

    private val visualizerReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            // Multiple safety checks for edge cases
            if (!isAdded || view == null || !isReceiverRegistered) return
            if (activity?.isFinishing == true || activity?.isDestroyed == true) return

            when (intent.action) {
                Constants.ACTION_VISUALIZER_DATA -> updateVisualization(intent)
                Constants.ACTION_FFT_DATA -> updateSpectrumAnalyzer(intent)
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.tab_dashboard, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Find views - Vulkan RT only
        vulkanSpectrumAnalyzer = view.findViewById(R.id.vulkan_spectrum_analyzer)
        chipVulkanRt = view.findViewById(R.id.chip_vulkan_rt)
        vuMeter = view.findViewById(R.id.vu_meter)
        statsMonitor = view.findViewById(R.id.stats_monitor)
        quickLatency = view.findViewById(R.id.quick_latency)
        quickLatencyStatus = view.findViewById(R.id.quick_latency_status)
        quickSampleRate = view.findViewById(R.id.quick_sample_rate)
        quickSampleStatus = view.findViewById(R.id.quick_sample_status)
        quickBuffer = view.findViewById(R.id.quick_buffer)
        quickBufferStatus = view.findViewById(R.id.quick_buffer_status)

        // Visual effects controls
        chipGroupTheme = view.findViewById(R.id.chip_group_theme)
        chipGroupStereo = view.findViewById(R.id.chip_group_stereo)

        // Configure Vulkan RT spectrum analyzer with default effects
        vulkanSpectrumAnalyzer?.apply {
            setGlowIntensity(1.2f)
            setBloomIntensity(0.6f)
            setChromaticAberration(0.15f)
            setScanlineIntensity(0.3f)
            setVignetteIntensity(0.5f)
        }

        // Setup theme selection listener
        chipGroupTheme?.setOnCheckedStateChangeListener { _, checkedIds ->
            if (checkedIds.isEmpty()) return@setOnCheckedStateChangeListener
            val theme = when (checkedIds.first()) {
                R.id.chip_theme_neon -> VulkanSpectrumView.THEME_NEON
                R.id.chip_theme_fire -> VulkanSpectrumView.THEME_FIRE
                R.id.chip_theme_matrix -> VulkanSpectrumView.THEME_MATRIX
                R.id.chip_theme_ocean -> VulkanSpectrumView.THEME_OCEAN
                else -> VulkanSpectrumView.THEME_NEON
            }
            vulkanSpectrumAnalyzer?.setColorTheme(theme)
            Timber.d("Theme changed to: $theme")
        }

        // Setup stereo mode selection listener
        chipGroupStereo?.setOnCheckedStateChangeListener { _, checkedIds ->
            if (checkedIds.isEmpty()) return@setOnCheckedStateChangeListener
            val mode = when (checkedIds.first()) {
                R.id.chip_stereo_mono -> VulkanSpectrumView.STEREO_MONO
                R.id.chip_stereo_mirror -> VulkanSpectrumView.STEREO_MIRROR
                R.id.chip_stereo_split -> VulkanSpectrumView.STEREO_SPLIT
                else -> VulkanSpectrumView.STEREO_MONO
            }
            vulkanSpectrumAnalyzer?.setStereoMode(mode)
            Timber.d("Stereo mode changed to: $mode")
        }

        setIdleState()

        // Update RT badge when ready - use safe delayed execution
        val rtBadgeRunnable = Runnable {
            if (!isAdded || view == null || activity?.isDestroyed == true) return@Runnable
            updateRtBadge()
        }
        pendingRunnables.add(rtBadgeRunnable)
        view.postDelayed(rtBadgeRunnable, 2000)
    }

    private fun updateRtBadge() {
        val ctx = context ?: return
        val vulkanReady = vulkanSpectrumAnalyzer?.isVulkanReady() ?: false
        val hasRT = vulkanSpectrumAnalyzer?.hasHardwareRayTracing() ?: false
        val rtMode = vulkanSpectrumAnalyzer?.getRayTracingMode() ?: "UNKNOWN"

        when {
            hasRT && rtMode == "RAY QUERY" -> {
                chipVulkanRt?.text = "RT QUERY"
                chipVulkanRt?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_success))
                chipVulkanRt?.setChipStrokeColorResource(R.color.ultra_success)
            }
            hasRT && rtMode == "FULL RT" -> {
                chipVulkanRt?.text = "FULL RT"
                chipVulkanRt?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_success))
                chipVulkanRt?.setChipStrokeColorResource(R.color.ultra_success)
            }
            vulkanReady -> {
                chipVulkanRt?.text = "VULKAN"
                chipVulkanRt?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_primary))
                chipVulkanRt?.setChipStrokeColorResource(R.color.ultra_primary)
            }
            else -> {
                chipVulkanRt?.text = "INIT..."
            }
        }
    }

    override fun onPause() {
        // Safely unregister receiver only if it was registered
        if (isReceiverRegistered) {
            try {
                context?.unregisterLocalReceiver(visualizerReceiver)
            } catch (e: Exception) {
                // Receiver was not registered, ignore
            }
            isReceiverRegistered = false
        }

        vuMeter?.reset()
        statsMonitor?.reset()
        vulkanSpectrumAnalyzer?.onPause()
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        vulkanSpectrumAnalyzer?.onResume()

        // Register receiver if not already registered
        if (!isReceiverRegistered) {
            context?.registerLocalReceiver(visualizerReceiver, IntentFilter().apply {
                addAction(Constants.ACTION_VISUALIZER_DATA)
                addAction(Constants.ACTION_FFT_DATA)
            })
            isReceiverRegistered = true
        }

        // Request initial data from service
        context?.sendLocalBroadcast(Intent(Constants.ACTION_VISUALIZER_REQUEST))
    }

    override fun onStop() {
        // In multi-window, onPause doesn't mean we're not visible
        // onStop is called when we're definitely not visible
        super.onStop()
    }

    private fun Context.sendLocalBroadcast(intent: Intent) {
        androidx.localbroadcastmanager.content.LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    override fun onDestroyView() {
        // Cancel all pending runnables to prevent callbacks to destroyed view
        view?.let { v ->
            pendingRunnables.forEach { v.removeCallbacks(it) }
        }
        pendingRunnables.clear()

        // Ensure receiver is unregistered
        if (isReceiverRegistered) {
            try {
                context?.unregisterLocalReceiver(visualizerReceiver)
            } catch (e: Exception) { }
            isReceiverRegistered = false
        }

        vulkanSpectrumAnalyzer = null
        chipVulkanRt = null
        vuMeter = null
        statsMonitor = null
        quickLatency = null
        quickLatencyStatus = null
        quickSampleRate = null
        quickSampleStatus = null
        quickBuffer = null
        quickBufferStatus = null
        chipGroupTheme = null
        chipGroupStereo = null
        super.onDestroyView()
    }

    private fun setIdleState() {
        quickLatency?.text = "--"
        quickLatencyStatus?.text = "IDLE"
        quickSampleRate?.text = "--"
        quickBuffer?.text = "--"
        vuMeter?.reset()
        vulkanSpectrumAnalyzer?.reset()
        statsMonitor?.reset()
    }

    private fun updateVisualization(intent: Intent) {
        val ctx = context ?: return

        // VU Meter data
        val leftRms = intent.getFloatExtra(Constants.EXTRA_VU_LEFT_RMS, -60f).sanitize(-60f)
        val rightRms = intent.getFloatExtra(Constants.EXTRA_VU_RIGHT_RMS, -60f).sanitize(-60f)
        val leftPeak = intent.getFloatExtra(Constants.EXTRA_VU_LEFT_PEAK, -60f).sanitize(-60f)
        val rightPeak = intent.getFloatExtra(Constants.EXTRA_VU_RIGHT_PEAK, -60f).sanitize(-60f)
        vuMeter?.updateLevels(leftRms, rightRms, leftPeak, rightPeak)

        // Stats data
        val latencyMs = intent.getFloatExtra(Constants.EXTRA_LATENCY_MS, 0f).sanitize(0f).coerceIn(0f, 1000f)
        val bufferFill = intent.getFloatExtra(Constants.EXTRA_BUFFER_FILL, 0f).sanitize(0f).coerceIn(0f, 1f)
        val throughput = intent.getFloatExtra(Constants.EXTRA_THROUGHPUT, 0f).sanitize(0f).coerceIn(0f, 10000f)
        val sampleRate = intent.getIntExtra(Constants.EXTRA_SAMPLE_RATE, 48000).coerceIn(8000, 384000)

        statsMonitor?.updateFromService(latencyMs, sampleRate, bufferFill, throughput)

        // Update quick stats
        quickLatency?.text = String.format("%.1f", latencyMs)
        quickLatencyStatus?.text = "ms"
        updateLatencyColor(latencyMs, ctx)

        quickSampleRate?.text = "${sampleRate / 1000}"
        quickSampleStatus?.text = "kHz"

        val healthPercent = (bufferFill * 100).toInt()
        quickBuffer?.text = "$healthPercent"
        quickBufferStatus?.text = "%"
        updateHealthColor(healthPercent.toFloat(), ctx)
    }

    private fun updateHealthColor(percent: Float, ctx: Context) {
        val color = when {
            percent >= 90f -> R.color.ultra_success
            percent >= 70f -> R.color.ultra_info
            percent >= 50f -> R.color.ultra_warning
            else -> R.color.ultra_error
        }
        quickBuffer?.setTextColor(ContextCompat.getColor(ctx, color))
    }

    private fun updateSpectrumAnalyzer(intent: Intent) {
        val isStereoMode = intent.getBooleanExtra(Constants.EXTRA_FFT_STEREO_MODE, false)

        if (isStereoMode) {
            val leftData = intent.getFloatArrayExtra(Constants.EXTRA_FFT_LEFT_MAGNITUDES)
            val rightData = intent.getFloatArrayExtra(Constants.EXTRA_FFT_RIGHT_MAGNITUDES)
            if (leftData != null && rightData != null) {
                vulkanSpectrumAnalyzer?.updateStereoFftData(leftData, rightData)
            }
        } else {
            val fftData = intent.getFloatArrayExtra(Constants.EXTRA_FFT_MAGNITUDES)
            if (fftData != null) {
                vulkanSpectrumAnalyzer?.updateFftData(fftData)
            }
        }
    }

    private fun updateLatencyColor(latencyMs: Float, ctx: Context) {
        val color = when {
            latencyMs < 5f -> R.color.ultra_success
            latencyMs < 10f -> R.color.ultra_info
            latencyMs < 20f -> R.color.ultra_warning
            else -> R.color.ultra_error
        }
        quickLatency?.setTextColor(ContextCompat.getColor(ctx, color))
    }

    private fun Float.sanitize(default: Float): Float {
        return if (this.isNaN() || this.isInfinite()) default else this
    }

    companion object {
        fun newInstance() = DashboardTabFragment()
    }
}
