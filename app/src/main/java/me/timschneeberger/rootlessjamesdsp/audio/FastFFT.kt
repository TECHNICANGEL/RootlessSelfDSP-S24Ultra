package me.timschneeberger.rootlessjamesdsp.audio

import java.util.concurrent.ExecutorService
import java.util.concurrent.Future
import java.util.concurrent.Callable

/**
 * S24 ULTRA - FFT & UI Helpers
 * ALL DSP IN C (s24_ultra_dsp.c) - Kotlin is thin wrapper only
 */
object FastFFT {

    @Volatile
    private var initialized = false

    @Synchronized
    fun initLookupTables() {
        if (initialized) return
        S24UltraDsp.initLookupTables()
        initColorTables()
        initialized = true
    }

    fun preWarm() {
        initLookupTables()
    }

    // ═══════════════════════════════════════════════════════════════
    // FAST MATH - Simple inline (OK in Kotlin)
    // ═══════════════════════════════════════════════════════════════

    fun fastToDB(mag: Float): Float = if (mag > 0.00001f) 20f * kotlin.math.log10(mag) else -100f
    fun fastDBToLinear(db: Float): Float = Math.pow(10.0, db / 20.0).toFloat()
    fun fastSqrt(x: Float): Float = kotlin.math.sqrt(x)
    fun fastAbs(x: Float): Float = if (x < 0) -x else x
    fun fastExp(x: Float): Float = kotlin.math.exp(x)

    fun softClip(x: Float): Float = when {
        x > 0.9f -> 0.9f + (x - 0.9f) / (1f + (x - 0.9f) * 10f)
        x < -0.9f -> -0.9f + (x + 0.9f) / (1f + (-x - 0.9f) * 10f)
        else -> x
    }

    fun getDecay(timeMs: Float, sampleRate: Int = 48000): Float {
        return kotlin.math.exp(-1f / (timeMs * 0.001f * sampleRate))
    }

    // ═══════════════════════════════════════════════════════════════
    // FFT - Delegates to native C
    // ═══════════════════════════════════════════════════════════════

    fun computeMagnitude(samples: FloatArray, magnitudes: FloatArray) =
        S24UltraDsp.fftMagnitude(samples, magnitudes)

    fun computeMagnitudeWindowed(samples: FloatArray, magnitudes: FloatArray) {
        if (samples.isEmpty()) return
        val windowed = samples.copyOf()
        S24UltraDsp.applyHannWindow(windowed)
        S24UltraDsp.fftMagnitude(windowed, magnitudes)
    }

    fun fft(input: FloatArray, size: Int = input.size): FloatArray {
        // Edge case: need at least 2 samples for FFT
        if (input.size < 2 || size < 2) return FloatArray(0)

        val magnitudes = FloatArray(size / 2)
        val windowed = input.copyOf()
        S24UltraDsp.applyHannWindow(windowed)
        S24UltraDsp.fftMagnitude(windowed, magnitudes)
        val result = FloatArray(magnitudes.size * 2)
        S24UltraDsp.magnitudeToInterleaved(magnitudes, result)
        return result
    }

    fun fftMagnitude(input: FloatArray, size: Int = input.size): FloatArray {
        // Edge case: need at least 2 samples for FFT
        if (input.size < 2 || size < 2) return FloatArray(0)

        val magnitudes = FloatArray(size / 2)
        val windowed = input.copyOf()
        S24UltraDsp.applyHannWindow(windowed)
        S24UltraDsp.fftMagnitude(windowed, magnitudes)
        return magnitudes
    }

    fun computeStereoMagnitude(
        leftSamples: FloatArray,
        rightSamples: FloatArray,
        leftMagnitudes: FloatArray,
        rightMagnitudes: FloatArray
    ) {
        computeMagnitudeWindowed(leftSamples, leftMagnitudes)
        computeMagnitudeWindowed(rightSamples, rightMagnitudes)
    }

    fun computeStereoMagnitudeParallel(
        leftSamples: FloatArray,
        rightSamples: FloatArray,
        leftMagnitudes: FloatArray,
        rightMagnitudes: FloatArray,
        executor: ExecutorService
    ) {
        val leftFuture: Future<*> = executor.submit(Callable { computeMagnitudeWindowed(leftSamples, leftMagnitudes) })
        val rightFuture: Future<*> = executor.submit(Callable { computeMagnitudeWindowed(rightSamples, rightMagnitudes) })
        try { leftFuture.get(); rightFuture.get() }
        catch (e: Exception) { computeStereoMagnitude(leftSamples, rightSamples, leftMagnitudes, rightMagnitudes) }
    }

    fun computeStereoMagnitudeParallelWithLogScale(
        leftSamples: FloatArray,
        rightSamples: FloatArray,
        leftMagnitudes: FloatArray,
        rightMagnitudes: FloatArray,
        leftMagnitudesLog: FloatArray,
        rightMagnitudesLog: FloatArray,
        sampleRate: Int,
        executor: ExecutorService
    ) {
        val leftFuture: Future<*> = executor.submit(Callable {
            computeMagnitudeWindowed(leftSamples, leftMagnitudes)
            linearToLogScale(leftMagnitudes, leftMagnitudesLog, sampleRate)
        })
        val rightFuture: Future<*> = executor.submit(Callable {
            computeMagnitudeWindowed(rightSamples, rightMagnitudes)
            linearToLogScale(rightMagnitudes, rightMagnitudesLog, sampleRate)
        })
        try { leftFuture.get(); rightFuture.get() }
        catch (e: Exception) {
            computeMagnitudeWindowed(leftSamples, leftMagnitudes)
            computeMagnitudeWindowed(rightSamples, rightMagnitudes)
            linearToLogScale(leftMagnitudes, leftMagnitudesLog, sampleRate)
            linearToLogScale(rightMagnitudes, rightMagnitudesLog, sampleRate)
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // LOG SCALE - Native C
    // ═══════════════════════════════════════════════════════════════

    fun linearToLogScale(
        linearMagnitudes: FloatArray,
        logMagnitudes: FloatArray,
        sampleRate: Int,
        minFreq: Float = 20f,
        maxFreq: Float = 20000f
    ) {
        if (linearMagnitudes.isEmpty() || logMagnitudes.isEmpty()) return
        S24UltraDsp.linearToLog(linearMagnitudes, logMagnitudes, sampleRate, minFreq, maxFreq)
    }

    // ═══════════════════════════════════════════════════════════════
    // SMOOTHING - Native C
    // ═══════════════════════════════════════════════════════════════

    fun applySmoothing(current: FloatArray, target: FloatArray, attack: Float = 0.8f, release: Float = 0.3f) {
        if (current.size != target.size || current.isEmpty()) return
        S24UltraDsp.applySmoothing(current, target, attack, release)
    }

    // ═══════════════════════════════════════════════════════════════
    // COLOR TABLES - UI only (OK in Kotlin)
    // ═══════════════════════════════════════════════════════════════

    private const val COLOR_SIZE = 1000
    private val spectrumColors = IntArray(COLOR_SIZE)
    private val heatMapColors = IntArray(COLOR_SIZE)

    private fun initColorTables() {
        for (i in 0 until COLOR_SIZE) {
            val t = i.toFloat() / COLOR_SIZE
            val (sr, sg, sb) = when {
                t < 0.25f -> Triple(0, (t / 0.25f * 255).toInt(), 255)
                t < 0.5f -> Triple(0, 255, ((1 - (t - 0.25f) / 0.25f) * 255).toInt())
                t < 0.75f -> Triple(((t - 0.5f) / 0.25f * 255).toInt(), 255, 0)
                else -> Triple(255, ((1 - (t - 0.75f) / 0.25f) * 255).toInt(), 0)
            }
            spectrumColors[i] = (0xFF shl 24) or (sr shl 16) or (sg shl 8) or sb

            val (hr, hg, hb) = when {
                t < 0.33f -> Triple((t / 0.33f * 255).toInt(), 0, 0)
                t < 0.66f -> Triple(255, ((t - 0.33f) / 0.33f * 255).toInt(), 0)
                else -> Triple(255, 255, ((t - 0.66f) / 0.34f * 255).toInt())
            }
            heatMapColors[i] = (0xFF shl 24) or (hr shl 16) or (hg shl 8) or hb
        }
    }

    fun getSpectrumColor(value: Float): Int = spectrumColors[(value.coerceIn(0f, 1f) * (COLOR_SIZE - 1)).toInt()]
    fun getHeatMapColor(value: Float): Int = heatMapColors[(value.coerceIn(0f, 1f) * (COLOR_SIZE - 1)).toInt()]
}
