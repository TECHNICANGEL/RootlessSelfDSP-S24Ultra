package me.timschneeberger.rootlessjamesdsp.view

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.AttributeSet
import android.view.Choreographer
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import timber.log.Timber
import java.util.concurrent.atomic.AtomicBoolean

/**
 * S24 ULTRA - Vulkan Spectrum Analyzer
 *
 * High-performance Vulkan renderer with:
 * - Hardware ray tracing on Adreno 750 (if available)
 * - 3D terrain deformation based on FFT spectrum
 * - Gyroscope-based parallax camera
 * - 120fps sustained rendering
 *
 * Uses native C++ Vulkan implementation for maximum performance
 */
class VulkanSpectrumView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : SurfaceView(context, attrs, defStyleAttr), SurfaceHolder.Callback, SensorEventListener {

    companion object {
        private const val TAG = "VulkanSpectrumView"
        private const val MIN_DB = -80f
        private const val NUM_BARS = 256

        // Color theme constants
        const val THEME_NEON = 0    // Purple/Cyan (default)
        const val THEME_FIRE = 1    // Orange/Red
        const val THEME_MATRIX = 2  // Green
        const val THEME_OCEAN = 3   // Blue/Teal

        // Stereo mode constants
        const val STEREO_MONO = 0     // Standard mono visualization
        const val STEREO_MIRROR = 1   // Left channel mirrored on left, right on right
        const val STEREO_SPLIT = 2    // Left channel on left half, right on right half

        init {
            try {
                System.loadLibrary("vulkan_spectrum")
                Timber.tag(TAG).i("vulkan_spectrum library loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Timber.tag(TAG).e(e, "Failed to load vulkan_spectrum library")
            }
        }
    }

    // Native handle
    private var nativeHandle: Long = 0

    // State flags
    private val surfaceReady = AtomicBoolean(false)
    private val isRendering = AtomicBoolean(false)
    private val isPaused = AtomicBoolean(true)
    private val hasRayTracing = AtomicBoolean(false)

    // Choreographer for vsync-aligned rendering (much more efficient than Handler.postDelayed)
    private var choreographer: Choreographer? = null

    // Buffered FFT data for when surface isn't ready yet
    @Volatile private var pendingFftData: FloatArray? = null
    @Volatile private var pendingLeftFftData: FloatArray? = null
    @Volatile private var pendingRightFftData: FloatArray? = null

    // FFT activity tracking - hide visualization when DSP is off
    @Volatile private var lastFftUpdateTime: Long = 0
    private val fftTimeoutMs: Long = 500  // Hide after 500ms without FFT data
    private val isActive = AtomicBoolean(false)

    // Gyroscope/accelerometer for parallax
    private var sensorManager: SensorManager? = null
    private var accelerometer: Sensor? = null
    @Volatile private var tiltX = 0f
    @Volatile private var tiltY = 0f
    // MEMORY LEAK FIX: Track sensor registration to prevent double registration
    private var isSensorRegistered = false

    // Choreographer frame callback - syncs to display refresh rate (60/120Hz)
    // Optimized: minimal allocations, no logging in hot path
    private val frameCallback = object : Choreographer.FrameCallback {
        private var lastVisibilityCheck = 0L
        private var surfaceStable = false

        override fun doFrame(frameTimeNanos: Long) {
            if (!isRendering.get() || isPaused.get() || !surfaceReady.get()) {
                return
            }

            val handle = nativeHandle
            if (handle == 0L) return

            // Wait for surface to be stable before controlling visibility
            if (!surfaceStable) {
                surfaceStable = true
                lastVisibilityCheck = frameTimeNanos / 1_000_000L
            }

            // Check FFT activity every 100ms (performance optimization)
            val currentTime = frameTimeNanos / 1_000_000L
            if (currentTime - lastVisibilityCheck > 100) {
                lastVisibilityCheck = currentTime
                val shouldBeActive = (System.currentTimeMillis() - lastFftUpdateTime) < fftTimeoutMs
                isActive.set(shouldBeActive)
            }

            // Always render - shader shows black/minimal when no FFT data
            nativeRenderWithTilt(handle, tiltX, tiltY)

            // Schedule next frame
            choreographer?.postFrameCallback(this)
        }
    }

    init {
        Timber.tag(TAG).d("VulkanSpectrumView init")

        // Keep VISIBLE so surface gets created, we control rendering internally
        // Surface won't be created if view is INVISIBLE

        // Create native renderer
        nativeHandle = nativeCreate()
        Timber.tag(TAG).d("Native handle created: $nativeHandle")

        // Setup surface callback
        holder.addCallback(this)

        // Setup sensors
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        accelerometer = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PUBLIC API
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Update FFT data from audio service (mono mode)
     * Thread-safe, can be called from any thread
     */
    fun updateFftData(data: FloatArray) {
        // Record FFT update time to track DSP activity
        lastFftUpdateTime = System.currentTimeMillis()

        if (nativeHandle != 0L && surfaceReady.get() && !isPaused.get()) {
            nativeUpdateFft(nativeHandle, data)
            pendingFftData = null  // Clear pending since we applied it
        } else if (!isPaused.get()) {
            // Buffer for when surface becomes ready
            pendingFftData = data.copyOf()
            pendingLeftFftData = null
            pendingRightFftData = null
        }
    }

    /**
     * Update stereo FFT data
     */
    fun updateStereoFftData(left: FloatArray, right: FloatArray) {
        lastFftUpdateTime = System.currentTimeMillis()

        if (nativeHandle != 0L && surfaceReady.get() && !isPaused.get()) {
            nativeUpdateStereoFft(nativeHandle, left, right)
            pendingLeftFftData = null
            pendingRightFftData = null
        } else if (!isPaused.get()) {
            pendingLeftFftData = left.copyOf()
            pendingRightFftData = right.copyOf()
            pendingFftData = null
        }
    }

    /**
     * Apply any buffered FFT data - called when surface becomes ready
     */
    private fun applyPendingFftData() {
        if (nativeHandle == 0L || !surfaceReady.get()) return

        pendingFftData?.let { data ->
            nativeUpdateFft(nativeHandle, data)
            pendingFftData = null
            Timber.tag(TAG).d("Applied pending mono FFT data")
        }

        val left = pendingLeftFftData
        val right = pendingRightFftData
        if (left != null && right != null) {
            nativeUpdateStereoFft(nativeHandle, left, right)
            pendingLeftFftData = null
            pendingRightFftData = null
            Timber.tag(TAG).d("Applied pending stereo FFT data")
        }
    }

    /**
     * Set glow effect intensity
     */
    fun setGlowIntensity(intensity: Float) {
        if (nativeHandle != 0L) {
            nativeSetGlowIntensity(nativeHandle, intensity)
        }
    }

    /**
     * Enable/disable glow effect
     */
    fun setGlowEnabled(enabled: Boolean) {
        setGlowIntensity(if (enabled) 1.0f else 0.0f)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VISUAL EFFECTS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Set bloom intensity (0.0 - 2.0)
     */
    fun setBloomIntensity(intensity: Float) {
        if (nativeHandle != 0L) {
            nativeSetBloomIntensity(nativeHandle, intensity.coerceIn(0f, 2f))
        }
    }

    /**
     * Set chromatic aberration amount (0.0 - 1.0)
     */
    fun setChromaticAberration(amount: Float) {
        if (nativeHandle != 0L) {
            nativeSetChromaticAberration(nativeHandle, amount.coerceIn(0f, 1f))
        }
    }

    /**
     * Set color theme
     * 0 = Neon (purple/cyan)
     * 1 = Fire (orange/red)
     * 2 = Matrix (green)
     * 3 = Ocean (blue/teal)
     */
    fun setColorTheme(theme: Int) {
        if (nativeHandle != 0L) {
            nativeSetColorTheme(nativeHandle, theme.coerceIn(0, 3))
        }
    }

    /**
     * Set scanline intensity (0.0 - 1.0)
     */
    fun setScanlineIntensity(intensity: Float) {
        if (nativeHandle != 0L) {
            nativeSetScanlineIntensity(nativeHandle, intensity.coerceIn(0f, 1f))
        }
    }

    /**
     * Set vignette intensity (0.0 - 1.0)
     */
    fun setVignetteIntensity(intensity: Float) {
        if (nativeHandle != 0L) {
            nativeSetVignetteIntensity(nativeHandle, intensity.coerceIn(0f, 1f))
        }
    }

    /**
     * Set stereo visualization mode
     * STEREO_MONO (0) = Standard mono visualization
     * STEREO_MIRROR (1) = Left channel mirrored on left, right on right
     * STEREO_SPLIT (2) = Left channel on left half, right on right half
     */
    fun setStereoMode(mode: Int) {
        if (nativeHandle != 0L) {
            nativeSetStereoMode(nativeHandle, mode.coerceIn(0, 2))
        }
    }

    /**
     * Check if hardware ray tracing is available (full pipeline or ray query)
     */
    fun hasHardwareRayTracing(): Boolean = hasRayTracing.get()

    /**
     * Get ray tracing mode string for display
     */
    fun getRayTracingMode(): String {
        return when {
            nativeHandle != 0L && nativeHasRayTracing(nativeHandle) -> {
                if (nativeHasRayQuery(nativeHandle)) "RAY QUERY" else "FULL RT"
            }
            else -> "SOFTWARE"
        }
    }

    /**
     * Check if Vulkan is ready
     */
    fun isVulkanReady(): Boolean = surfaceReady.get() && nativeHandle != 0L

    /**
     * Reset to idle state
     */
    fun reset() {
        if (nativeHandle != 0L) {
            nativeReset(nativeHandle)
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SURFACE CALLBACKS
    // ═══════════════════════════════════════════════════════════════════════════

    override fun surfaceCreated(holder: SurfaceHolder) {
        // Check if native handle is valid (may have been destroyed)
        val handle = nativeHandle
        if (handle == 0L) return

        val surface = holder.surface
        if (surface.isValid) {
            val success = nativeInit(handle, surface)
            if (success) {
                surfaceReady.set(true)
                hasRayTracing.set(nativeHasRayTracing(handle))

                // Apply any FFT data that arrived before surface was ready
                applyPendingFftData()

                // Start Choreographer rendering if not paused
                if (!isPaused.get()) {
                    startRendering()
                }
            }
        }
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        // Vulkan handles swapchain recreation internally
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        surfaceReady.set(false)
        stopRendering()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════════

    fun onResume() {
        Timber.tag(TAG).d("onResume: nativeHandle=$nativeHandle, surfaceReady=${surfaceReady.get()}, isRendering=${isRendering.get()}")
        isPaused.set(false)

        // Reset FFT activity state - start inactive until FFT data arrives
        isActive.set(false)
        lastFftUpdateTime = 0

        // Register sensors (only if not already registered)
        if (!isSensorRegistered) {
            accelerometer?.let {
                sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
                isSensorRegistered = true
            }
        }

        // Resume native renderer
        if (nativeHandle != 0L) {
            nativeResume(nativeHandle)
        }

        // Force restart rendering if surface is ready
        // Reset isRendering to ensure startRendering actually posts the callback
        if (surfaceReady.get()) {
            isRendering.set(false) // Force restart
            startRendering()
        }
    }

    fun onPause() {
        isPaused.set(true)

        // Clear pending FFT data to avoid stale visualization on resume
        pendingFftData = null
        pendingLeftFftData = null
        pendingRightFftData = null

        // Unregister sensors
        sensorManager?.unregisterListener(this)
        isSensorRegistered = false

        // Pause native renderer
        if (nativeHandle != 0L) {
            nativePause(nativeHandle)
        }

        stopRendering()
    }

    override fun onVisibilityChanged(changedView: View, visibility: Int) {
        super.onVisibilityChanged(changedView, visibility)
        when (visibility) {
            VISIBLE -> {
                if (!isPaused.get() && surfaceReady.get()) {
                    startRendering()
                }
            }
            INVISIBLE, GONE -> {
                stopRendering()
            }
        }
    }

    override fun onWindowVisibilityChanged(visibility: Int) {
        super.onWindowVisibilityChanged(visibility)
        when (visibility) {
            VISIBLE -> {
                if (!isPaused.get() && surfaceReady.get()) {
                    startRendering()
                }
            }
            INVISIBLE, GONE -> {
                stopRendering()
            }
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        Timber.tag(TAG).d("onAttachedToWindow: nativeHandle=$nativeHandle, surfaceReady=${surfaceReady.get()}")

        // Re-initialize native renderer if it was destroyed
        if (nativeHandle == 0L) {
            nativeHandle = nativeCreate()
            Timber.tag(TAG).d("Re-created native handle: $nativeHandle")
        }

        // Re-add surface callback if needed
        holder.addCallback(this)

        // Re-setup sensors
        if (sensorManager == null) {
            sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
            accelerometer = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        }

        // Get choreographer instance
        if (choreographer == null) {
            choreographer = Choreographer.getInstance()
        }

        // CRITICAL: If surface already exists but surfaceReady is false, re-initialize
        // This handles the case where the view is re-attached but surfaceCreated() won't be called
        val surface = holder.surface
        if (surface != null && surface.isValid && !surfaceReady.get() && nativeHandle != 0L) {
            Timber.tag(TAG).d("Surface already valid on attach, re-initializing Vulkan")
            val success = nativeInit(nativeHandle, surface)
            if (success) {
                surfaceReady.set(true)
                hasRayTracing.set(nativeHasRayTracing(nativeHandle))
                applyPendingFftData()
                // Don't start rendering here - wait for onResume()
            }
        }
    }

    override fun onDetachedFromWindow() {
        Timber.tag(TAG).d("onDetachedFromWindow")
        surfaceReady.set(false)
        stopRendering()

        // Remove surface callback to prevent memory leaks
        holder.removeCallback(this)

        // Unregister sensors to prevent leaks
        sensorManager?.unregisterListener(this)
        isSensorRegistered = false

        // Clear pending data
        pendingFftData = null
        pendingLeftFftData = null
        pendingRightFftData = null

        // Destroy native renderer
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0
        }

        choreographer = null
        super.onDetachedFromWindow()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CHOREOGRAPHER RENDERING (vsync-aligned, more efficient than HandlerThread)
    // ═══════════════════════════════════════════════════════════════════════════

    private fun startRendering() {
        // Always try to start rendering - remove callback first to prevent duplicates
        if (isRendering.get()) {
            // Already rendering, but ensure callback is scheduled
            return
        }

        Timber.tag(TAG).d("Starting Choreographer rendering")
        isRendering.set(true)

        // Get or create choreographer on main thread
        if (choreographer == null) {
            choreographer = Choreographer.getInstance()
        }

        // Remove any pending callback first to prevent duplicates
        choreographer?.removeFrameCallback(frameCallback)
        choreographer?.postFrameCallback(frameCallback)
    }

    private fun stopRendering() {
        Timber.tag(TAG).d("Stopping Choreographer rendering, wasRendering=${isRendering.get()}")
        isRendering.set(false)
        choreographer?.removeFrameCallback(frameCallback)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SENSOR LISTENER - Gyroscope parallax
    // ═══════════════════════════════════════════════════════════════════════════

    // Optimized sensor callback - minimal operations
    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ACCELEROMETER) return

        val v0 = event.values[0]
        val v1 = event.values[1]

        // Quick NaN check
        if (v0 != v0 || v1 != v1) return  // NaN != NaN is true

        // Simplified low-pass filter directly to tilt
        val alpha = 0.02f
        val invGravity = 0.102f  // 1/9.8

        tiltX += alpha * ((v0 * invGravity).coerceIn(-1f, 1f) - tiltX)
        tiltY += alpha * ((v1 * invGravity).coerceIn(-1f, 1f) - tiltY)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // NATIVE METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    private external fun nativeCreate(): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeInit(handle: Long, surface: Surface): Boolean
    private external fun nativeRender(handle: Long)
    private external fun nativeRenderWithTilt(handle: Long, tiltX: Float, tiltY: Float)  // Combined for fewer JNI calls
    private external fun nativePause(handle: Long)
    private external fun nativeResume(handle: Long)
    private external fun nativeUpdateFft(handle: Long, data: FloatArray)
    private external fun nativeUpdateStereoFft(handle: Long, left: FloatArray, right: FloatArray)
    private external fun nativeSetTilt(handle: Long, x: Float, y: Float)
    private external fun nativeSetGlowIntensity(handle: Long, intensity: Float)
    private external fun nativeReset(handle: Long)
    private external fun nativeHasRayTracing(handle: Long): Boolean
    private external fun nativeHasRayQuery(handle: Long): Boolean
    // Visual effects
    private external fun nativeSetBloomIntensity(handle: Long, intensity: Float)
    private external fun nativeSetChromaticAberration(handle: Long, amount: Float)
    private external fun nativeSetColorTheme(handle: Long, theme: Int)
    private external fun nativeSetScanlineIntensity(handle: Long, intensity: Float)
    private external fun nativeSetVignetteIntensity(handle: Long, intensity: Float)
    private external fun nativeSetStereoMode(handle: Long, mode: Int)
}
