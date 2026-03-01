package me.timschneeberger.rootlessjamesdsp.audio

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES31
import timber.log.Timber
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * S24 ULTRA - GPU Compute for Audio DSP
 * OpenGL ES 3.1 Compute Shaders on Adreno 750
 *
 * Offloads parallelizable operations:
 * - Large FFT magnitude computation
 * - Spectral smoothing
 * - Peak detection across bins
 * - Batch audio analysis
 *
 * Thread-safe: Uses lock to ensure only one thread accesses EGL context at a time
 * (EGL contexts can only be current on one thread)
 */
object GpuCompute {

    private const val TAG = "GpuCompute"
    private const val WORKGROUP_SIZE = 256 // Optimal for Adreno 750
    private const val EGL_OPENGL_ES3_BIT_KHR = 0x0040 // Not in EGL14, define manually

    private val initialized = AtomicBoolean(false)
    private val available = AtomicBoolean(false)

    // EGL contexts can only be current on one thread at a time
    // Use a lock to serialize access across threads
    private val eglLock = ReentrantLock()

    @Volatile private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    @Volatile private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    @Volatile private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE

    // Compute shader programs
    private var smoothingProgram = 0
    private var peakProgram = 0
    private var magnitudeProgram = 0
    private var spectralProgram = 0

    // SSBOs (Shader Storage Buffer Objects)
    private var inputBuffer = 0
    private var outputBuffer = 0
    private var paramsBuffer = 0

    // Pre-allocated CPU buffers
    private var cpuInputBuffer: FloatBuffer? = null
    private var cpuOutputBuffer: FloatBuffer? = null
    private const val MAX_BUFFER_SIZE = 32768

    fun initialize(): Boolean {
        if (initialized.get()) return available.get()

        return eglLock.withLock {
            // Double-check after acquiring lock
            if (initialized.get()) return@withLock available.get()

            try {
                if (!initEGL()) {
                    Timber.tag(TAG).w("EGL initialization failed")
                    initialized.set(true)
                    return@withLock false
                }

                if (!checkComputeSupport()) {
                    Timber.tag(TAG).w("Compute shaders not supported")
                    releaseEGL()
                    initialized.set(true)
                    return@withLock false
                }

                createBuffers()
                compileShaders()

                // Release context from init thread - will be acquired per-operation
                releaseCurrent()

                available.set(true)
                initialized.set(true)
                Timber.tag(TAG).i("GPU Compute initialized - Adreno 750 ready")
                return@withLock true

            } catch (e: Exception) {
                Timber.tag(TAG).e(e, "GPU Compute initialization failed")
                initialized.set(true)
                return@withLock false
            }
        }
    }

    private fun initEGL(): Boolean {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) return false

        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) return false

        val configAttribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL14.EGL_NONE
        )

        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)
        if (numConfigs[0] == 0) return false

        val contextAttribs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
            EGL14.EGL_NONE
        )

        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
        if (eglContext == EGL14.EGL_NO_CONTEXT) return false

        val surfaceAttribs = intArrayOf(
            EGL14.EGL_WIDTH, 1,
            EGL14.EGL_HEIGHT, 1,
            EGL14.EGL_NONE
        )

        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, configs[0], surfaceAttribs, 0)
        if (eglSurface == EGL14.EGL_NO_SURFACE) return false

        return EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)
    }

    private fun checkComputeSupport(): Boolean {
        val version = GLES31.glGetString(GLES31.GL_VERSION) ?: return false
        Timber.tag(TAG).i("OpenGL: $version")

        val maxWorkGroupCount = IntArray(3)
        val maxWorkGroupSize = IntArray(3)
        GLES31.glGetIntegeri_v(GLES31.GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, maxWorkGroupCount, 0)
        GLES31.glGetIntegeri_v(GLES31.GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, maxWorkGroupSize, 0)

        Timber.tag(TAG).i("Max work groups: ${maxWorkGroupCount[0]}, Max work group size: ${maxWorkGroupSize[0]}")
        return maxWorkGroupSize[0] >= WORKGROUP_SIZE
    }

    private fun createBuffers() {
        val buffers = IntArray(3)
        GLES31.glGenBuffers(3, buffers, 0)
        inputBuffer = buffers[0]
        outputBuffer = buffers[1]
        paramsBuffer = buffers[2]

        // Pre-allocate GPU buffers
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, inputBuffer)
        GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, MAX_BUFFER_SIZE * 4, null, GLES31.GL_DYNAMIC_DRAW)

        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, outputBuffer)
        GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, MAX_BUFFER_SIZE * 4, null, GLES31.GL_DYNAMIC_DRAW)

        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, paramsBuffer)
        GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, 16 * 4, null, GLES31.GL_DYNAMIC_DRAW)

        // CPU buffers for transfer
        cpuInputBuffer = ByteBuffer.allocateDirect(MAX_BUFFER_SIZE * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        cpuOutputBuffer = ByteBuffer.allocateDirect(MAX_BUFFER_SIZE * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
    }

    private fun compileShaders() {
        smoothingProgram = createComputeProgram(SMOOTHING_SHADER)
        peakProgram = createComputeProgram(PEAK_SHADER)
        magnitudeProgram = createComputeProgram(MAGNITUDE_SHADER)
        spectralProgram = createComputeProgram(SPECTRAL_SHADER)
    }

    private fun createComputeProgram(source: String): Int {
        val shader = GLES31.glCreateShader(GLES31.GL_COMPUTE_SHADER)
        GLES31.glShaderSource(shader, source)
        GLES31.glCompileShader(shader)

        val compiled = IntArray(1)
        GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES31.glGetShaderInfoLog(shader)
            Timber.tag(TAG).e("Shader compile error: $log")
            GLES31.glDeleteShader(shader)
            return 0
        }

        val program = GLES31.glCreateProgram()
        GLES31.glAttachShader(program, shader)
        GLES31.glLinkProgram(program)

        val linked = IntArray(1)
        GLES31.glGetProgramiv(program, GLES31.GL_LINK_STATUS, linked, 0)
        if (linked[0] == 0) {
            val log = GLES31.glGetProgramInfoLog(program)
            Timber.tag(TAG).e("Program link error: $log")
            GLES31.glDeleteProgram(program)
            return 0
        }

        GLES31.glDeleteShader(shader)
        return program
    }

    // ═══════════════════════════════════════════════════════════════
    // PUBLIC API - GPU Compute Operations
    // ═══════════════════════════════════════════════════════════════

    /**
     * GPU-accelerated smoothing for large arrays (4096+ elements)
     * Falls back to CPU for small arrays or if GPU is unavailable/busy
     */
    fun smoothing(current: FloatArray, target: FloatArray, attack: Float, release: Float) {
        if (!available.get() || current.size < 1024) {
            S24UltraDsp.applySmoothing(current, target, attack, release)
            return
        }

        // Try to acquire lock without blocking to avoid audio thread stalls
        if (!eglLock.tryLock()) {
            // Another thread is using GPU, fall back to CPU
            S24UltraDsp.applySmoothing(current, target, attack, release)
            return
        }

        try {
            if (!makeCurrent()) {
                // EGL context unavailable, fall back to CPU
                S24UltraDsp.applySmoothing(current, target, attack, release)
                return
            }

            // Upload data
            cpuInputBuffer?.clear()
            cpuInputBuffer?.put(current)
            cpuInputBuffer?.put(target)
            cpuInputBuffer?.flip()

            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, inputBuffer)
            GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, current.size * 4 * 2, cpuInputBuffer)

            // Upload params
            val params = floatArrayOf(attack, release, current.size.toFloat(), 0f)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, paramsBuffer)
            GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, 16, FloatBuffer.wrap(params))

            // Dispatch
            GLES31.glUseProgram(smoothingProgram)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, inputBuffer)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, outputBuffer)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, paramsBuffer)

            val workGroups = (current.size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
            GLES31.glDispatchCompute(workGroups, 1, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)

            // Download result via buffer mapping
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, outputBuffer)
            val mappedBuffer = GLES31.glMapBufferRange(
                GLES31.GL_SHADER_STORAGE_BUFFER, 0, current.size * 4,
                GLES31.GL_MAP_READ_BIT
            ) as? ByteBuffer
            mappedBuffer?.order(ByteOrder.nativeOrder())?.asFloatBuffer()?.get(current)
            GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)

            releaseCurrent()
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "GPU smoothing failed, falling back to CPU")
            S24UltraDsp.applySmoothing(current, target, attack, release)
        } finally {
            eglLock.unlock()
        }
    }

    /**
     * GPU-accelerated peak detection with reduction
     * Falls back to CPU if GPU is unavailable/busy
     */
    fun findPeak(samples: FloatArray): Float {
        if (!available.get() || samples.size < 4096) {
            return S24UltraDsp.findPeak(samples)
        }

        // Try to acquire lock without blocking to avoid audio thread stalls
        if (!eglLock.tryLock()) {
            return S24UltraDsp.findPeak(samples)
        }

        try {
            if (!makeCurrent()) {
                return S24UltraDsp.findPeak(samples)
            }

            // Upload data
            cpuInputBuffer?.clear()
            cpuInputBuffer?.put(samples)
            cpuInputBuffer?.flip()

            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, inputBuffer)
            GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, samples.size * 4, cpuInputBuffer)

            // Params
            val params = floatArrayOf(samples.size.toFloat(), 0f, 0f, 0f)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, paramsBuffer)
            GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, 16, FloatBuffer.wrap(params))

            // Dispatch
            GLES31.glUseProgram(peakProgram)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, inputBuffer)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, outputBuffer)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, paramsBuffer)

            val workGroups = (samples.size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
            GLES31.glDispatchCompute(workGroups, 1, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)

            // Get partial results via buffer mapping and reduce on CPU
            val partialPeaks = FloatArray(workGroups)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, outputBuffer)
            val mappedBuffer = GLES31.glMapBufferRange(
                GLES31.GL_SHADER_STORAGE_BUFFER, 0, workGroups * 4,
                GLES31.GL_MAP_READ_BIT
            ) as? ByteBuffer
            mappedBuffer?.order(ByteOrder.nativeOrder())?.asFloatBuffer()?.get(partialPeaks)
            GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)

            releaseCurrent()
            return partialPeaks.maxOrNull() ?: 0f
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "GPU findPeak failed, falling back to CPU")
            return S24UltraDsp.findPeak(samples)
        } finally {
            eglLock.unlock()
        }
    }

    /**
     * GPU-accelerated spectral analysis
     * Computes centroid, flux, bass/mid/high energy in parallel
     * Falls back to CPU if GPU is unavailable/busy
     */
    fun spectralAnalysis(
        magnitudes: FloatArray,
        previousMagnitudes: FloatArray,
        sampleRate: Int,
        fftSize: Int
    ): SpectralResult {
        if (!available.get() || magnitudes.size < 2048) {
            return cpuSpectralAnalysis(magnitudes, previousMagnitudes, sampleRate, fftSize)
        }

        // Try to acquire lock without blocking to avoid audio thread stalls
        if (!eglLock.tryLock()) {
            return cpuSpectralAnalysis(magnitudes, previousMagnitudes, sampleRate, fftSize)
        }

        try {
            if (!makeCurrent()) {
                return cpuSpectralAnalysis(magnitudes, previousMagnitudes, sampleRate, fftSize)
            }

            // Upload magnitudes and previous
            cpuInputBuffer?.clear()
            cpuInputBuffer?.put(magnitudes)
            cpuInputBuffer?.put(previousMagnitudes)
            cpuInputBuffer?.flip()

            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, inputBuffer)
            GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, magnitudes.size * 4 * 2, cpuInputBuffer)

            // Params: size, sampleRate, fftSize, bassEnd, midEnd
            val bassEnd = fftSize / 32
            val midEnd = fftSize / 4
            val params = floatArrayOf(
                magnitudes.size.toFloat(),
                sampleRate.toFloat(),
                fftSize.toFloat(),
                bassEnd.toFloat(),
                midEnd.toFloat(),
                0f, 0f, 0f
            )
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, paramsBuffer)
            GLES31.glBufferSubData(GLES31.GL_SHADER_STORAGE_BUFFER, 0, 32, FloatBuffer.wrap(params))

            // Dispatch
            GLES31.glUseProgram(spectralProgram)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, inputBuffer)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, outputBuffer)
            GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 2, paramsBuffer)

            val workGroups = (magnitudes.size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
            GLES31.glDispatchCompute(workGroups, 1, 1)
            GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT)

            // Download partial results via buffer mapping (per workgroup: centroidNum, centroidDen, flux, maxMag, sumMag, bass, mid, high)
            val resultsPerGroup = 8
            val partialResults = FloatArray(workGroups * resultsPerGroup)
            GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, outputBuffer)
            val mappedBuffer = GLES31.glMapBufferRange(
                GLES31.GL_SHADER_STORAGE_BUFFER, 0, partialResults.size * 4,
                GLES31.GL_MAP_READ_BIT
            ) as? ByteBuffer
            mappedBuffer?.order(ByteOrder.nativeOrder())?.asFloatBuffer()?.get(partialResults)
            GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER)

            releaseCurrent()

            // Final reduction on CPU
            var centroidNum = 0f
            var centroidDen = 0f
            var flux = 0f
            var maxMag = 0f
            var sumMag = 0f
            var bassSum = 0f
            var midSum = 0f
            var highSum = 0f

            for (g in 0 until workGroups) {
                val base = g * resultsPerGroup
                centroidNum += partialResults[base]
                centroidDen += partialResults[base + 1]
                flux += partialResults[base + 2]
                if (partialResults[base + 3] > maxMag) maxMag = partialResults[base + 3]
                sumMag += partialResults[base + 4]
                bassSum += partialResults[base + 5]
                midSum += partialResults[base + 6]
                highSum += partialResults[base + 7]
            }

            // Update previousMagnitudes (copy current to previous)
            System.arraycopy(magnitudes, 0, previousMagnitudes, 0, magnitudes.size)

            return SpectralResult(
                centroid = if (centroidDen > 0) centroidNum / centroidDen else 0f,
                flux = flux,
                crestFactor = if (sumMag > 0) maxMag / (sumMag / magnitudes.size) else 0f,
                bassEnergy = bassSum,
                midEnergy = midSum,
                highEnergy = highSum
            )
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "GPU spectralAnalysis failed, falling back to CPU")
            return cpuSpectralAnalysis(magnitudes, previousMagnitudes, sampleRate, fftSize)
        } finally {
            eglLock.unlock()
        }
    }

    private fun cpuSpectralAnalysis(
        magnitudes: FloatArray,
        previousMagnitudes: FloatArray,
        sampleRate: Int,
        fftSize: Int
    ): SpectralResult {
        var centroidNum = 0f
        var centroidDen = 0f
        var flux = 0f
        var maxMag = 0f
        var sumMag = 0f
        var bassSum = 0f
        var midSum = 0f
        var highSum = 0f

        val bassEnd = fftSize / 32
        val midEnd = fftSize / 4

        for (i in magnitudes.indices) {
            val mag = magnitudes[i]
            val freq = i.toFloat() * sampleRate / fftSize

            centroidNum += freq * mag
            centroidDen += mag
            flux += kotlin.math.abs(mag - previousMagnitudes[i])
            previousMagnitudes[i] = mag
            if (mag > maxMag) maxMag = mag
            sumMag += mag

            when {
                i < bassEnd -> bassSum += mag
                i < midEnd -> midSum += mag
                else -> highSum += mag
            }
        }

        return SpectralResult(
            centroid = if (centroidDen > 0) centroidNum / centroidDen else 0f,
            flux = flux,
            crestFactor = if (sumMag > 0) maxMag / (sumMag / magnitudes.size) else 0f,
            bassEnergy = bassSum,
            midEnergy = midSum,
            highEnergy = highSum
        )
    }

    data class SpectralResult(
        val centroid: Float,
        val flux: Float,
        val crestFactor: Float,
        val bassEnergy: Float,
        val midEnergy: Float,
        val highEnergy: Float
    )

    /**
     * Make EGL context current on this thread.
     * Must be called while holding eglLock.
     * Returns false if EGL resources are invalid or makeCurrent fails.
     */
    private fun makeCurrent(): Boolean {
        // Validate EGL resources exist
        if (eglDisplay == EGL14.EGL_NO_DISPLAY ||
            eglContext == EGL14.EGL_NO_CONTEXT ||
            eglSurface == EGL14.EGL_NO_SURFACE) {
            Timber.tag(TAG).w("makeCurrent: EGL resources not initialized")
            return false
        }

        // Attempt to make current
        val success = EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)
        if (!success) {
            val error = EGL14.eglGetError()
            // Only log if not EGL_BAD_ACCESS (common during shutdown/thread contention)
            if (error != 0x3002) { // EGL_BAD_ACCESS
                Timber.tag(TAG).w("makeCurrent failed: error 0x%04X", error)
            }
            return false
        }
        return true
    }

    /**
     * Release EGL context from current thread.
     * Should be called after GPU operations complete.
     */
    private fun releaseCurrent() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        }
    }

    private fun releaseEGL() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            if (eglContext != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(eglDisplay, eglContext)
            if (eglSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(eglDisplay, eglSurface)
            EGL14.eglTerminate(eglDisplay)
        }
        eglDisplay = EGL14.EGL_NO_DISPLAY
        eglContext = EGL14.EGL_NO_CONTEXT
        eglSurface = EGL14.EGL_NO_SURFACE
    }

    fun release() {
        if (!initialized.get()) return

        eglLock.withLock {
            // Mark as unavailable first to stop new operations
            available.set(false)

            if (makeCurrent()) {
                if (smoothingProgram != 0) GLES31.glDeleteProgram(smoothingProgram)
                if (peakProgram != 0) GLES31.glDeleteProgram(peakProgram)
                if (magnitudeProgram != 0) GLES31.glDeleteProgram(magnitudeProgram)
                if (spectralProgram != 0) GLES31.glDeleteProgram(spectralProgram)

                val buffers = intArrayOf(inputBuffer, outputBuffer, paramsBuffer)
                GLES31.glDeleteBuffers(3, buffers, 0)
            }

            releaseEGL()

            cpuInputBuffer = null
            cpuOutputBuffer = null

            smoothingProgram = 0
            peakProgram = 0
            magnitudeProgram = 0
            spectralProgram = 0
            inputBuffer = 0
            outputBuffer = 0
            paramsBuffer = 0

            initialized.set(false)

            Timber.tag(TAG).i("GPU Compute released")
        }
    }

    fun isAvailable() = available.get()

    // ═══════════════════════════════════════════════════════════════
    // COMPUTE SHADERS - GLSL ES 3.1
    // ═══════════════════════════════════════════════════════════════

    private const val SMOOTHING_SHADER = """#version 310 es
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer InputBuffer {
    float data[];  // current followed by target
};

layout(std430, binding = 1) buffer OutputBuffer {
    float result[];
};

layout(std430, binding = 2) buffer Params {
    float attack;
    float release;
    float size;
    float padding;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint n = uint(size);
    if (idx >= n) return;

    float current = data[idx];
    float target = data[idx + n];

    if (target > current) {
        result[idx] = current * (1.0 - attack) + target * attack;
    } else {
        result[idx] = current * release + target * (1.0 - release);
    }
}
"""

    private const val PEAK_SHADER = """#version 310 es
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer InputBuffer {
    float samples[];
};

layout(std430, binding = 1) buffer OutputBuffer {
    float partialPeaks[];  // One per workgroup
};

layout(std430, binding = 2) buffer Params {
    float size;
    float pad1, pad2, pad3;
};

shared float sharedPeaks[256];

void main() {
    uint globalIdx = gl_GlobalInvocationID.x;
    uint localIdx = gl_LocalInvocationID.x;
    uint groupIdx = gl_WorkGroupID.x;
    uint n = uint(size);

    // Load and take absolute value
    float val = 0.0;
    if (globalIdx < n) {
        val = abs(samples[globalIdx]);
    }
    sharedPeaks[localIdx] = val;

    memoryBarrierShared();
    barrier();

    // Parallel reduction within workgroup
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (localIdx < stride) {
            sharedPeaks[localIdx] = max(sharedPeaks[localIdx], sharedPeaks[localIdx + stride]);
        }
        memoryBarrierShared();
        barrier();
    }

    // Write workgroup result
    if (localIdx == 0u) {
        partialPeaks[groupIdx] = sharedPeaks[0];
    }
}
"""

    private const val MAGNITUDE_SHADER = """#version 310 es
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer InputBuffer {
    float realImag[];  // Interleaved real/imag
};

layout(std430, binding = 1) buffer OutputBuffer {
    float magnitudes[];
};

layout(std430, binding = 2) buffer Params {
    float size;  // Number of complex samples
    float pad1, pad2, pad3;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint n = uint(size);
    if (idx >= n) return;

    float real = realImag[idx * 2u];
    float imag = realImag[idx * 2u + 1u];
    magnitudes[idx] = sqrt(real * real + imag * imag);
}
"""

    private const val SPECTRAL_SHADER = """#version 310 es
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer InputBuffer {
    float data[];  // magnitudes followed by previousMagnitudes
};

layout(std430, binding = 1) buffer OutputBuffer {
    float results[];  // Per workgroup: centroidNum, centroidDen, flux, maxMag, sumMag, bass, mid, high
};

layout(std430, binding = 2) buffer Params {
    float size;
    float sampleRate;
    float fftSize;
    float bassEnd;
    float midEnd;
    float pad1, pad2, pad3;
};

shared float sharedCentroidNum[256];
shared float sharedCentroidDen[256];
shared float sharedFlux[256];
shared float sharedMaxMag[256];
shared float sharedSumMag[256];
shared float sharedBass[256];
shared float sharedMid[256];
shared float sharedHigh[256];

void main() {
    uint globalIdx = gl_GlobalInvocationID.x;
    uint localIdx = gl_LocalInvocationID.x;
    uint groupIdx = gl_WorkGroupID.x;
    uint n = uint(size);

    float centroidNum = 0.0;
    float centroidDen = 0.0;
    float flux = 0.0;
    float maxMag = 0.0;
    float sumMag = 0.0;
    float bass = 0.0;
    float mid = 0.0;
    float high = 0.0;

    if (globalIdx < n) {
        float mag = data[globalIdx];
        float prevMag = data[globalIdx + n];
        float freq = float(globalIdx) * sampleRate / fftSize;

        centroidNum = freq * mag;
        centroidDen = mag;
        flux = abs(mag - prevMag);
        maxMag = mag;
        sumMag = mag;

        if (float(globalIdx) < bassEnd) {
            bass = mag;
        } else if (float(globalIdx) < midEnd) {
            mid = mag;
        } else {
            high = mag;
        }
    }

    sharedCentroidNum[localIdx] = centroidNum;
    sharedCentroidDen[localIdx] = centroidDen;
    sharedFlux[localIdx] = flux;
    sharedMaxMag[localIdx] = maxMag;
    sharedSumMag[localIdx] = sumMag;
    sharedBass[localIdx] = bass;
    sharedMid[localIdx] = mid;
    sharedHigh[localIdx] = high;

    memoryBarrierShared();
    barrier();

    // Parallel reduction
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (localIdx < stride) {
            sharedCentroidNum[localIdx] += sharedCentroidNum[localIdx + stride];
            sharedCentroidDen[localIdx] += sharedCentroidDen[localIdx + stride];
            sharedFlux[localIdx] += sharedFlux[localIdx + stride];
            sharedMaxMag[localIdx] = max(sharedMaxMag[localIdx], sharedMaxMag[localIdx + stride]);
            sharedSumMag[localIdx] += sharedSumMag[localIdx + stride];
            sharedBass[localIdx] += sharedBass[localIdx + stride];
            sharedMid[localIdx] += sharedMid[localIdx + stride];
            sharedHigh[localIdx] += sharedHigh[localIdx + stride];
        }
        memoryBarrierShared();
        barrier();
    }

    // Write workgroup results
    if (localIdx == 0u) {
        uint base = groupIdx * 8u;
        results[base] = sharedCentroidNum[0];
        results[base + 1u] = sharedCentroidDen[0];
        results[base + 2u] = sharedFlux[0];
        results[base + 3u] = sharedMaxMag[0];
        results[base + 4u] = sharedSumMag[0];
        results[base + 5u] = sharedBass[0];
        results[base + 6u] = sharedMid[0];
        results[base + 7u] = sharedHigh[0];
    }
}
"""
}
