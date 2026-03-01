package me.timschneeberger.rootlessjamesdsp.audio

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.Spatializer
import android.os.Build
import timber.log.Timber
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * S24 ULTRA - Spatial Audio with Head Tracking
 *
 * Tries multiple head tracking sources in order:
 * 1. Android 13+ TYPE_HEAD_TRACKER (from XM5 via Bluetooth)
 * 2. Android Spatializer API head tracker
 * 3. Phone's gyroscope (fallback)
 *
 * Converts stereo audio to binaural 3D using HRTF convolution.
 */
class SpatialAudio(private val context: Context) : SensorEventListener {

    companion object {
        private const val TAG = "SpatialAudio"
        private const val DEFAULT_SAMPLE_RATE = 48000
        private const val SENSOR_DELAY = SensorManager.SENSOR_DELAY_GAME

        // Sensor.TYPE_HEAD_TRACKER = 37 (added in API 33)
        private const val TYPE_HEAD_TRACKER = 37

        // Track if native library is loaded
        private var nativeLibraryLoaded = false

        init {
            try {
                System.loadLibrary("npu_audio")
                nativeLibraryLoaded = true
                Timber.tag(TAG).i("npu_audio library loaded")
            } catch (e: UnsatisfiedLinkError) {
                nativeLibraryLoaded = false
                Timber.tag(TAG).w("npu_audio library not available - Spatial Audio disabled")
            }
        }

        fun isNativeLibraryAvailable() = nativeLibraryLoaded
    }

    private var nativeHandle: Long = 0
    private val isInitialized = AtomicBoolean(false)
    private val headTrackingEnabled = AtomicBoolean(false)

    // Sensor management
    private var sensorManager: SensorManager? = null
    private var audioManager: AudioManager? = null
    private var spatializer: Spatializer? = null

    // Head tracker sources (in priority order)
    private var headTrackerSensor: Sensor? = null      // TYPE_HEAD_TRACKER from XM5
    private var rotationSensor: Sensor? = null          // Phone rotation vector
    private var accelerometer: Sensor? = null           // Phone accelerometer
    private var magnetometer: Sensor? = null            // Phone magnetometer

    // Which source is active
    enum class HeadTrackingSource {
        NONE, HEADPHONE_SENSOR, SPATIALIZER_API, PHONE_GYRO
    }
    private val activeSource = AtomicReference(HeadTrackingSource.NONE)

    // Head orientation (degrees)
    private val azimuth = AtomicReference(0f)    // Horizontal: -180 to 180
    private val elevation = AtomicReference(0f)  // Vertical: -90 to 90
    private val roll = AtomicReference(0f)       // Tilt: -180 to 180

    // Sensor fusion data
    private val rotationMatrix = FloatArray(9)
    private val orientationAngles = FloatArray(3)
    private var lastAccelerometer = FloatArray(3)
    private var lastMagnetometer = FloatArray(3)
    private var hasAccelerometer = false
    private var hasMagnetometer = false

    // Reference orientation (set when enabling head tracking)
    private var referenceAzimuth = 0f
    private var referenceElevation = 0f

    // Processing buffers
    private var leftOutBuffer = FloatArray(0)
    private var rightOutBuffer = FloatArray(0)

    // PERFORMANCE: Buffer pools for processInterleaved (avoids GC in hot path)
    private var deinterleaveLeftPool: AudioBufferPool.FloatBufferPool? = null
    private var deinterleaveRightPool: AudioBufferPool.FloatBufferPool? = null
    private var interleavedOutputPool: AudioBufferPool.FloatBufferPool? = null
    private var lastFrameCount = 0
    private var lastStereoSize = 0

    // Effect intensity (0.0 = off, 1.0 = full spatial)
    private val intensity = AtomicReference(0.7f)

    fun initialize(sampleRate: Int = DEFAULT_SAMPLE_RATE): Boolean {
        if (isInitialized.get()) return true

        // Check if native library is available
        if (!nativeLibraryLoaded) {
            Timber.tag(TAG).w("Cannot initialize - native library not loaded")
            return false
        }

        return try {
            nativeHandle = nativeInit(sampleRate)
            if (nativeHandle != 0L) {
                isInitialized.set(true)
                initializeSensors()
                initializeSpatializer()
                logCapabilities()
                Timber.tag(TAG).i("Spatial audio initialized (sample rate: $sampleRate)")
                true
            } else {
                Timber.tag(TAG).w("Spatial audio init failed")
                false
            }
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Spatial audio initialization failed")
            false
        }
    }

    private fun initializeSensors() {
        // Use applicationContext to avoid leaking BroadcastReceiver when Service is destroyed
        // getDynamicSensorList() internally registers a receiver that won't be cleaned up
        // if using a Service context
        val appContext = context.applicationContext
        sensorManager = appContext.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        audioManager = appContext.getSystemService(Context.AUDIO_SERVICE) as? AudioManager

        sensorManager?.let { sm ->
            // 1. Try TYPE_HEAD_TRACKER (Android 13+, from Bluetooth headphones)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                headTrackerSensor = sm.getDefaultSensor(TYPE_HEAD_TRACKER)
                if (headTrackerSensor != null) {
                    Timber.tag(TAG).i("✅ HEAD_TRACKER sensor found: ${headTrackerSensor?.name}")
                } else {
                    // Try to find it in dynamic sensors list
                    // Note: getDynamicSensorList() registers an internal BroadcastReceiver
                    // Using applicationContext prevents this from being leaked
                    try {
                        val dynamicSensors = sm.getDynamicSensorList(TYPE_HEAD_TRACKER)
                        if (dynamicSensors.isNotEmpty()) {
                            headTrackerSensor = dynamicSensors[0]
                            Timber.tag(TAG).i("✅ HEAD_TRACKER found in dynamic sensors: ${headTrackerSensor?.name}")
                        } else {
                            Timber.tag(TAG).w("❌ HEAD_TRACKER sensor not available")
                        }
                    } catch (e: Exception) {
                        Timber.tag(TAG).w("Failed to get dynamic sensors: ${e.message}")
                    }
                }
            }

            // 2. Phone sensors as fallback
            rotationSensor = sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            if (rotationSensor != null) {
                Timber.tag(TAG).i("📱 Rotation vector sensor available")
            } else {
                accelerometer = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
                magnetometer = sm.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
                Timber.tag(TAG).i("📱 Using accelerometer + magnetometer fallback")
            }

            // Log all available sensors for debugging
            logAvailableSensors(sm)
        }
    }

    private fun logAvailableSensors(sm: SensorManager) {
        Timber.tag(TAG).d("=== Available Sensors ===")

        // Check for head tracking related sensors
        val sensorTypes = listOf(
            TYPE_HEAD_TRACKER to "HEAD_TRACKER",
            Sensor.TYPE_ROTATION_VECTOR to "ROTATION_VECTOR",
            Sensor.TYPE_GAME_ROTATION_VECTOR to "GAME_ROTATION_VECTOR",
            Sensor.TYPE_GYROSCOPE to "GYROSCOPE",
            Sensor.TYPE_ACCELEROMETER to "ACCELEROMETER"
        )

        for ((type, name) in sensorTypes) {
            val sensor = sm.getDefaultSensor(type)
            if (sensor != null) {
                Timber.tag(TAG).d("  ✓ $name: ${sensor.name} (${sensor.vendor})")
            } else {
                Timber.tag(TAG).d("  ✗ $name: not available")
            }
        }

        // Check dynamic sensors (Bluetooth headphones)
        // Note: getDynamicSensorList registers an internal BroadcastReceiver
        // This is safe because we use applicationContext in initializeSensors()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            try {
                val dynamicSensors = sm.getDynamicSensorList(Sensor.TYPE_ALL)
                if (dynamicSensors.isNotEmpty()) {
                    Timber.tag(TAG).d("=== Dynamic Sensors (Bluetooth) ===")
                    for (sensor in dynamicSensors) {
                        Timber.tag(TAG).d("  ✓ ${sensor.stringType}: ${sensor.name} (${sensor.vendor})")
                    }
                } else {
                    Timber.tag(TAG).d("No dynamic sensors found")
                }
            } catch (e: Exception) {
                Timber.tag(TAG).d("Failed to enumerate dynamic sensors: ${e.message}")
            }
        }
    }

    private fun initializeSpatializer() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            try {
                spatializer = audioManager?.spatializer
                spatializer?.let { sp ->
                    val level = sp.immersiveAudioLevel
                    Timber.tag(TAG).i("Spatializer level: $level")
                    Timber.tag(TAG).i("Spatializer enabled: ${sp.isEnabled}")
                    Timber.tag(TAG).i("Spatializer available: ${sp.isAvailable}")

                    // Check if head tracking is available
                    val isHeadTrackerAvailable = try {
                        sp.isHeadTrackerAvailable
                    } catch (e: Exception) {
                        Timber.tag(TAG).w("isHeadTrackerAvailable check failed: ${e.message}")
                        false
                    }
                    Timber.tag(TAG).i("Head tracker available: $isHeadTrackerAvailable")

                    // Check if current audio can be spatialized
                    val audioAttributes = AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build()

                    val audioFormat = AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setSampleRate(48000)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                        .build()

                    val canBeSpatilized = sp.canBeSpatialized(audioAttributes, audioFormat)
                    Timber.tag(TAG).i("Can be spatialized: $canBeSpatilized")
                }
            } catch (e: Exception) {
                Timber.tag(TAG).w(e, "Spatializer initialization failed")
            }
        }
    }

    private fun logCapabilities() {
        Timber.tag(TAG).i("""
            === Spatial Audio Capabilities ===
            Head Tracker Sensor: ${headTrackerSensor != null}
            Spatializer API: ${spatializer != null}
            Phone Rotation: ${rotationSensor != null}
            Phone Accel+Mag: ${accelerometer != null && magnetometer != null}
        """.trimIndent())
    }

    /**
     * Enable head tracking - tries best available source
     */
    fun enableHeadTracking() {
        if (!isInitialized.get()) return

        sensorManager?.let { sm ->
            // Priority 1: Headphone head tracker (XM5)
            headTrackerSensor?.let { sensor ->
                if (sm.registerListener(this, sensor, SENSOR_DELAY)) {
                    activeSource.set(HeadTrackingSource.HEADPHONE_SENSOR)
                    headTrackingEnabled.set(true)
                    Timber.tag(TAG).i("🎧 Head tracking enabled via HEADPHONE SENSOR")
                    setReferenceOrientation()
                    return
                }
            }

            // Priority 2: Phone rotation vector
            rotationSensor?.let { sensor ->
                if (sm.registerListener(this, sensor, SENSOR_DELAY)) {
                    activeSource.set(HeadTrackingSource.PHONE_GYRO)
                    headTrackingEnabled.set(true)
                    Timber.tag(TAG).i("📱 Head tracking enabled via PHONE GYRO")
                    setReferenceOrientation()
                    return
                }
            }

            // Priority 3: Accelerometer + Magnetometer
            val accelOk = accelerometer?.let { sm.registerListener(this, it, SENSOR_DELAY) } ?: false
            val magOk = magnetometer?.let { sm.registerListener(this, it, SENSOR_DELAY) } ?: false

            if (accelOk && magOk) {
                activeSource.set(HeadTrackingSource.PHONE_GYRO)
                headTrackingEnabled.set(true)
                Timber.tag(TAG).i("📱 Head tracking enabled via ACCEL+MAG")
                setReferenceOrientation()
                return
            }

            Timber.tag(TAG).w("❌ No head tracking source available")
            activeSource.set(HeadTrackingSource.NONE)
        }
    }

    fun disableHeadTracking() {
        sensorManager?.unregisterListener(this)
        headTrackingEnabled.set(false)
        activeSource.set(HeadTrackingSource.NONE)

        // Reset to center
        azimuth.set(0f)
        elevation.set(0f)
        roll.set(0f)

        // Only call native if library is loaded and initialized
        if (nativeLibraryLoaded && nativeHandle != 0L) {
            nativeUpdateHeadOrientation(nativeHandle, 0f, 0f)
        }

        Timber.tag(TAG).i("Head tracking disabled")
    }

    private fun setReferenceOrientation() {
        referenceAzimuth = azimuth.get()
        referenceElevation = elevation.get()
    }

    /**
     * Reset reference position (re-center)
     */
    fun recenter() {
        setReferenceOrientation()
        Timber.tag(TAG).d("Head tracking recentered")
    }

    // Processing lock for thread safety
    private val processLock = Any()

    /**
     * Process stereo audio with spatial rendering
     * Thread-safe with proper edge case handling
     */
    fun process(left: FloatArray, right: FloatArray): Pair<FloatArray, FloatArray> {
        // Edge case: not initialized
        if (!isInitialized.get() || nativeHandle == 0L) {
            return Pair(left, right)
        }

        // Edge case: empty arrays
        if (left.isEmpty() || right.isEmpty()) {
            return Pair(left, right)
        }

        // Edge case: mismatched sizes
        if (left.size != right.size) {
            Timber.tag(TAG).w("Mismatched L/R sizes: ${left.size} vs ${right.size}")
            return Pair(left, right)
        }

        synchronized(processLock) {
            try {
                if (leftOutBuffer.size != left.size) {
                    leftOutBuffer = FloatArray(left.size)
                    rightOutBuffer = FloatArray(right.size)
                }

                nativeProcess(nativeHandle, left, right, leftOutBuffer, rightOutBuffer)

                // Sanitize output for NaN/Infinity
                for (i in leftOutBuffer.indices) {
                    if (leftOutBuffer[i].isNaN() || leftOutBuffer[i].isInfinite()) {
                        leftOutBuffer[i] = left[i]
                    }
                    if (rightOutBuffer[i].isNaN() || rightOutBuffer[i].isInfinite()) {
                        rightOutBuffer[i] = right[i]
                    }
                }

                return Pair(leftOutBuffer.copyOf(), rightOutBuffer.copyOf())
            } catch (e: Exception) {
                Timber.tag(TAG).e(e, "Native process failed")
                return Pair(left, right)
            }
        }
    }

    /**
     * Process interleaved stereo
     * Thread-safe with proper edge case handling
     */
    fun processInterleaved(stereo: FloatArray): FloatArray {
        // Edge case: not initialized
        if (!isInitialized.get() || nativeHandle == 0L) return stereo

        // Edge case: empty or odd-sized input
        if (stereo.isEmpty() || stereo.size < 2 || stereo.size % 2 != 0) {
            return stereo
        }

        val frameCount = stereo.size / 2

        // PERFORMANCE: Use buffer pools for deinterleave buffers
        if (frameCount != lastFrameCount) {
            deinterleaveLeftPool = AudioBufferPool.getFloatPool(frameCount, 4)
            deinterleaveRightPool = AudioBufferPool.getFloatPool(frameCount, 4)
            lastFrameCount = frameCount
        }
        if (stereo.size != lastStereoSize) {
            interleavedOutputPool = AudioBufferPool.getFloatPool(stereo.size, 4)
            lastStereoSize = stereo.size
        }

        val left = deinterleaveLeftPool?.acquire() ?: FloatArray(frameCount)
        val right = deinterleaveRightPool?.acquire() ?: FloatArray(frameCount)

        // Deinterleave with bounds checking
        for (i in 0 until frameCount) {
            val stereoIdx = i * 2
            if (stereoIdx + 1 < stereo.size) {
                left[i] = stereo[stereoIdx].sanitize(0f)
                right[i] = stereo[stereoIdx + 1].sanitize(0f)
            }
        }

        val (outL, outR) = process(left, right)

        // PERFORMANCE: Release deinterleave buffers back to pool
        deinterleaveLeftPool?.release(left)
        deinterleaveRightPool?.release(right)

        // Edge case: process returned different sizes
        if (outL.size != frameCount || outR.size != frameCount) {
            return stereo
        }

        val output = interleavedOutputPool?.acquire() ?: FloatArray(stereo.size)
        for (i in 0 until frameCount) {
            output[i * 2] = outL[i]
            output[i * 2 + 1] = outR[i]
        }

        return output
    }

    /**
     * Release a buffer back to the interleaved output pool
     * Call this after you're done using the result from processInterleaved()
     */
    fun releaseBuffer(buffer: FloatArray) {
        interleavedOutputPool?.release(buffer)
    }

    /**
     * Sanitize float - replace NaN/Infinity with default
     */
    private fun Float.sanitize(default: Float): Float {
        return if (this.isNaN() || this.isInfinite()) default else this
    }

    // Sensor callbacks
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            TYPE_HEAD_TRACKER -> {
                // Head tracker provides quaternion or rotation vector
                handleHeadTrackerData(event)
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                SensorManager.getOrientation(rotationMatrix, orientationAngles)
                updateOrientation(orientationAngles)
            }
            Sensor.TYPE_ACCELEROMETER -> {
                System.arraycopy(event.values, 0, lastAccelerometer, 0, 3)
                hasAccelerometer = true
                updateOrientationFromAccelMag()
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                System.arraycopy(event.values, 0, lastMagnetometer, 0, 3)
                hasMagnetometer = true
                updateOrientationFromAccelMag()
            }
        }
    }

    private fun handleHeadTrackerData(event: SensorEvent) {
        // TYPE_HEAD_TRACKER reports discontinuous rotation as 6-axis
        // values[0-2]: rotation vector (rx, ry, rz)
        // values[3-5]: velocity (vx, vy, vz) - optional

        if (event.values.size >= 3) {
            // Convert rotation vector to euler angles
            val rx = event.values[0]
            val ry = event.values[1]
            val rz = event.values[2]

            // Rotation vector to quaternion
            val theta = kotlin.math.sqrt(rx * rx + ry * ry + rz * rz)
            val quaternion = if (theta > 0.0001f) {
                val halfTheta = theta / 2f
                val scale = kotlin.math.sin(halfTheta) / theta
                floatArrayOf(
                    kotlin.math.cos(halfTheta),
                    rx * scale,
                    ry * scale,
                    rz * scale
                )
            } else {
                floatArrayOf(1f, 0f, 0f, 0f)
            }

            // Quaternion to euler
            val q0 = quaternion[0]
            val q1 = quaternion[1]
            val q2 = quaternion[2]
            val q3 = quaternion[3]

            val sinYaw = 2.0f * (q0 * q3 + q1 * q2)
            val cosYaw = 1.0f - 2.0f * (q2 * q2 + q3 * q3)
            var yaw = kotlin.math.atan2(sinYaw, cosYaw)

            val sinPitch = 2.0f * (q0 * q2 - q3 * q1)
            var pitch = if (kotlin.math.abs(sinPitch) >= 1f) {
                Math.copySign(Math.PI.toFloat() / 2f, sinPitch)
            } else {
                kotlin.math.asin(sinPitch)
            }

            // Convert to degrees and apply reference
            yaw = Math.toDegrees(yaw.toDouble()).toFloat() - referenceAzimuth
            pitch = Math.toDegrees(pitch.toDouble()).toFloat() - referenceElevation

            // Normalize
            while (yaw > 180f) yaw -= 360f
            while (yaw < -180f) yaw += 360f
            pitch = pitch.coerceIn(-90f, 90f)

            azimuth.set(yaw)
            elevation.set(pitch)

            if (nativeLibraryLoaded && nativeHandle != 0L) {
                nativeUpdateHeadOrientation(nativeHandle, yaw, pitch)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Ignore
    }

    private fun updateOrientation(angles: FloatArray) {
        var az = Math.toDegrees(angles[0].toDouble()).toFloat()
        var el = Math.toDegrees(angles[1].toDouble()).toFloat()

        az -= referenceAzimuth
        el -= referenceElevation

        while (az > 180f) az -= 360f
        while (az < -180f) az += 360f
        el = el.coerceIn(-90f, 90f)

        azimuth.set(az)
        elevation.set(el)

        if (nativeLibraryLoaded && nativeHandle != 0L) {
            nativeUpdateHeadOrientation(nativeHandle, az, el)
        }
    }

    private fun updateOrientationFromAccelMag() {
        if (!hasAccelerometer || !hasMagnetometer) return

        val success = SensorManager.getRotationMatrix(
            rotationMatrix, null,
            lastAccelerometer, lastMagnetometer
        )

        if (success) {
            SensorManager.getOrientation(rotationMatrix, orientationAngles)
            updateOrientation(orientationAngles)
        }
    }

    // Settings
    fun setIntensity(value: Float) = intensity.set(value.coerceIn(0f, 1f))
    fun getIntensity() = intensity.get()
    fun getAzimuth() = azimuth.get()
    fun getElevation() = elevation.get()
    fun getRoll() = roll.get()
    fun isHeadTrackingEnabled() = headTrackingEnabled.get()
    fun getActiveSource() = activeSource.get()

    fun release() {
        if (!isInitialized.getAndSet(false)) return

        synchronized(processLock) {
            try {
                disableHeadTracking()
            } catch (e: Exception) {
                Timber.tag(TAG).e(e, "Error disabling head tracking")
            }

            if (nativeLibraryLoaded && nativeHandle != 0L) {
                try {
                    nativeRelease(nativeHandle)
                } catch (e: Exception) {
                    Timber.tag(TAG).e(e, "Error releasing native handle")
                }
                nativeHandle = 0
            }

            // Clear buffers
            leftOutBuffer = FloatArray(0)
            rightOutBuffer = FloatArray(0)

            // Clear sensor references
            sensorManager = null
            audioManager = null
            spatializer = null
            headTrackerSensor = null
            rotationSensor = null
            accelerometer = null
            magnetometer = null

            Timber.tag(TAG).i("Spatial audio released")
        }
    }

    fun isAvailable() = isInitialized.get()

    // Native methods
    private external fun nativeInit(sampleRate: Int): Long
    private external fun nativeUpdateHeadOrientation(handle: Long, azimuth: Float, elevation: Float)
    private external fun nativeProcess(
        handle: Long,
        leftIn: FloatArray, rightIn: FloatArray,
        leftOut: FloatArray, rightOut: FloatArray
    )
    private external fun nativeRelease(handle: Long)
}
