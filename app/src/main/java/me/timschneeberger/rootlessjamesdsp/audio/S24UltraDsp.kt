package me.timschneeberger.rootlessjamesdsp.audio

import timber.log.Timber

/**
 * ███████╗██████╗ ██╗  ██╗    ██╗   ██╗██╗  ████████╗██████╗  █████╗
 * ██╔════╝╚════██╗██║  ██║    ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔══██╗
 * ███████╗ █████╔╝███████║    ██║   ██║██║     ██║   ██████╔╝███████║
 * ╚════██║██╔═══╝ ╚════██║    ██║   ██║██║     ██║   ██╔══██╗██╔══██║
 * ███████║███████╗     ██║    ╚██████╔╝███████╗██║   ██║  ██║██║  ██║
 * ╚══════╝╚══════╝     ╚═╝     ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
 *
 * SAMSUNG GALAXY S24 ULTRA - Native DSP Library
 * ALL DSP IN C - NO KOTLIN FALLBACKS
 *
 * Snapdragon 8 Gen 3 / Cortex-X4 @ 3.3GHz:
 * - 4x 128-bit NEON execution pipes
 * - 64-byte cache lines
 * - FMA single cycle
 */
object S24UltraDsp {

    init {
        System.loadLibrary("s24_ultra_dsp")
        Timber.i("S24 Ultra DSP: Native library loaded")
    }

    // ═══════════════════════════════════════════════════════════════
    // NATIVE METHODS - ALL DSP IN C
    // ═══════════════════════════════════════════════════════════════

    external fun isS24UltraOptimized(): Boolean

    // Basic DSP
    private external fun nativeApplyGain(samples: FloatArray, gain: Float)
    private external fun nativeFindPeak(samples: FloatArray): Float
    private external fun nativeCalculateRms(samples: FloatArray): Float
    private external fun nativeStereoDeinterleave(interleaved: FloatArray, left: FloatArray, right: FloatArray)
    private external fun nativeStereoInterleave(left: FloatArray, right: FloatArray, interleaved: FloatArray)
    private external fun nativeStereoWidth(left: FloatArray, right: FloatArray, width: Float)
    private external fun nativeMixBuffers(src1: FloatArray, gain1: Float, src2: FloatArray, gain2: Float, dst: FloatArray)
    private external fun nativeCrossfeed(left: FloatArray, right: FloatArray, amount: Float)
    private external fun nativeBalance(left: FloatArray, right: FloatArray, pan: Float)
    private external fun nativeSoftClip(samples: FloatArray)
    private external fun nativeHardClip(samples: FloatArray, threshold: Float)
    private external fun nativePeakStereo(left: FloatArray, right: FloatArray): FloatArray
    private external fun nativeRmsStereo(left: FloatArray, right: FloatArray): FloatArray
    private external fun nativeFftMagnitude(input: FloatArray, magnitudes: FloatArray)
    private external fun nativeApplyHannWindow(samples: FloatArray)

    // Lookup tables
    private external fun nativeInitLookupTables()

    // Dynamics
    private external fun nativeCompressor(samples: FloatArray, threshold: Float, ratio: Float,
                                          attack: Float, release: Float, makeup: Float, sampleRate: Int)
    private external fun nativeLimiter(samples: FloatArray, threshold: Float, release: Float, sampleRate: Int)
    private external fun nativeGate(samples: FloatArray, threshold: Float, attack: Float,
                                    hold: Float, release: Float, range: Float, sampleRate: Int)
    private external fun nativeExpander(samples: FloatArray, threshold: Float, ratio: Float,
                                        attack: Float, release: Float, sampleRate: Int)
    private external fun nativeResetDynamics()

    // Biquad filters
    private external fun nativeBiquadCreate(): Int
    private external fun nativeBiquadPeakEq(id: Int, freq: Float, gainDb: Float, q: Float, sampleRate: Int)
    private external fun nativeBiquadLowShelf(id: Int, freq: Float, gainDb: Float, q: Float, sampleRate: Int)
    private external fun nativeBiquadHighShelf(id: Int, freq: Float, gainDb: Float, q: Float, sampleRate: Int)
    private external fun nativeBiquadLowpass(id: Int, freq: Float, q: Float, sampleRate: Int)
    private external fun nativeBiquadHighpass(id: Int, freq: Float, q: Float, sampleRate: Int)
    private external fun nativeBiquadBandpass(id: Int, freq: Float, q: Float, sampleRate: Int)
    private external fun nativeBiquadNotch(id: Int, freq: Float, q: Float, sampleRate: Int)
    private external fun nativeBiquadProcess(id: Int, left: FloatArray, right: FloatArray)
    private external fun nativeBiquadReset(id: Int)
    private external fun nativeBiquadResetAll()

    // LUFS metering
    private external fun nativeLufsInit(sampleRate: Int)
    private external fun nativeLufsProcess(left: FloatArray, right: FloatArray)
    private external fun nativeLufsGet(): FloatArray
    private external fun nativeLufsReset()

    // DC removal
    private external fun nativeRemoveDc(left: FloatArray, right: FloatArray)

    // Analysis (moved from Kotlin)
    private external fun nativeAnalyzeStereoInterleaved(buffer: FloatArray): FloatArray
    private external fun nativeAnalyzeBlock(samples: FloatArray): FloatArray
    private external fun nativeStereoCorrelation(left: FloatArray, right: FloatArray): Float
    private external fun nativeSpectralFeatures(samples: FloatArray): FloatArray
    private external fun nativeSpectralSubtract(frame: FloatArray, noiseProfile: FloatArray, reduction: Float)
    private external fun nativeLinearToLog(linear: FloatArray, logOut: FloatArray, sampleRate: Int, minFreq: Float, maxFreq: Float)
    private external fun nativeApplySmoothing(current: FloatArray, target: FloatArray, attack: Float, release: Float)
    private external fun nativeSpatialize(left: FloatArray, right: FloatArray, width: Float)
    private external fun nativeEnhanceContent(samples: FloatArray, contentType: Int)
    private external fun nativeUpdateNoiseFloor(silence: FloatArray, noiseFloor: FloatArray, smoothing: Float)

    // Conversion functions
    private external fun nativeShortToFloat(src: ShortArray, dst: FloatArray)
    private external fun nativeFloatToShort(src: FloatArray, dst: ShortArray)
    private external fun nativeMagnitudeToInterleaved(magnitudes: FloatArray, interleaved: FloatArray)

    // Additional DSP
    private external fun nativeDeesser(samples: FloatArray, thresholdDb: Float, sampleRate: Int)
    private external fun nativeEnhanceBass(samples: FloatArray, amount: Float, sampleRate: Int)
    private external fun nativeExciter(samples: FloatArray, amount: Float)
    private external fun nativeRestoreClips(samples: FloatArray)
    private external fun nativeNormalizeLoudness(samples: FloatArray, targetLufs: Float)
    private external fun nativeStereoToMono(left: FloatArray, right: FloatArray, mono: FloatArray)
    private external fun nativeIntegratedLoudness(history: FloatArray): Float
    private external fun nativeDownsampleFft(input: FloatArray, output: FloatArray, smoothingFactor: Float)
    private external fun nativeDeinterleaveToCircular(interleaved: FloatArray, leftAcc: FloatArray, rightAcc: FloatArray, position: Int, accSize: Int, frameCount: Int)
    private external fun nativeCircularToLinear(circular: FloatArray, linear: FloatArray, position: Int, circSize: Int)
    private external fun nativePeakHoldDecay(current: FloatArray, peakHold: FloatArray, decayRate: Float)
    private external fun nativeAverageMagnitudes(left: FloatArray, right: FloatArray, output: FloatArray)
    private external fun nativeDeinterleaveShortToCircular(interleaved: ShortArray, leftAcc: FloatArray, rightAcc: FloatArray, position: Int, accSize: Int, frameCount: Int)
    private external fun nativeAnalyzeStereoShort(buffer: ShortArray): LongArray

    // Complete processing chain
    private external fun nativeProcessChain(
        samples: FloatArray,
        applyEq: Boolean, eqFilterIds: IntArray?, numEqBands: Int,
        applyCompressor: Boolean, compThreshold: Float, compRatio: Float,
        compAttack: Float, compRelease: Float, compMakeup: Float,
        applyLimiter: Boolean, limitThreshold: Float,
        applyStereoWidth: Boolean, width: Float,
        applyCrossfeed: Boolean, crossfeedAmount: Float,
        sampleRate: Int
    )

    // ═══════════════════════════════════════════════════════════════
    // PUBLIC API - Direct native calls, no fallbacks
    // ═══════════════════════════════════════════════════════════════

    private var lookupTablesInitialized = false

    fun initLookupTables() {
        if (!lookupTablesInitialized) {
            nativeInitLookupTables()
            lookupTablesInitialized = true
        }
    }

    // Basic DSP
    fun applyGain(samples: FloatArray, gain: Float) = nativeApplyGain(samples, gain)
    fun findPeak(samples: FloatArray): Float = nativeFindPeak(samples)
    fun calculateRms(samples: FloatArray): Float = nativeCalculateRms(samples)
    fun stereoDeinterleave(interleaved: FloatArray, left: FloatArray, right: FloatArray) = nativeStereoDeinterleave(interleaved, left, right)
    fun stereoInterleave(left: FloatArray, right: FloatArray, interleaved: FloatArray) = nativeStereoInterleave(left, right, interleaved)
    fun stereoWidth(left: FloatArray, right: FloatArray, width: Float) = nativeStereoWidth(left, right, width)
    fun mixBuffers(src1: FloatArray, gain1: Float, src2: FloatArray, gain2: Float, dst: FloatArray) = nativeMixBuffers(src1, gain1, src2, gain2, dst)
    fun crossfeed(left: FloatArray, right: FloatArray, amount: Float) = nativeCrossfeed(left, right, amount)
    fun balance(left: FloatArray, right: FloatArray, pan: Float) = nativeBalance(left, right, pan)
    fun softClip(samples: FloatArray) = nativeSoftClip(samples)
    fun hardClip(samples: FloatArray, threshold: Float = 1f) = nativeHardClip(samples, threshold)
    fun peakStereo(left: FloatArray, right: FloatArray): Pair<Float, Float> {
        val r = nativePeakStereo(left, right)
        return Pair(r[0], r[1])
    }
    fun rmsStereo(left: FloatArray, right: FloatArray): Pair<Float, Float> {
        val r = nativeRmsStereo(left, right)
        return Pair(r[0], r[1])
    }
    fun fftMagnitude(input: FloatArray, magnitudes: FloatArray) = nativeFftMagnitude(input, magnitudes)
    fun applyHannWindow(samples: FloatArray) = nativeApplyHannWindow(samples)

    // Dynamics
    fun compressor(samples: FloatArray, thresholdDb: Float, ratio: Float, attackMs: Float, releaseMs: Float, makeupDb: Float, sampleRate: Int) =
        nativeCompressor(samples, thresholdDb, ratio, attackMs, releaseMs, makeupDb, sampleRate)
    fun limiter(samples: FloatArray, thresholdDb: Float, releaseMs: Float, sampleRate: Int) =
        nativeLimiter(samples, thresholdDb, releaseMs, sampleRate)
    fun gate(samples: FloatArray, thresholdDb: Float, attackMs: Float, holdMs: Float, releaseMs: Float, rangeDb: Float, sampleRate: Int) =
        nativeGate(samples, thresholdDb, attackMs, holdMs, releaseMs, rangeDb, sampleRate)
    fun expander(samples: FloatArray, thresholdDb: Float, ratio: Float, attackMs: Float, releaseMs: Float, sampleRate: Int) =
        nativeExpander(samples, thresholdDb, ratio, attackMs, releaseMs, sampleRate)
    fun resetDynamics() = nativeResetDynamics()

    // Biquad filters
    fun biquadCreate(): Int = nativeBiquadCreate()
    fun biquadPeakEq(id: Int, freq: Float, gainDb: Float, q: Float, sampleRate: Int) = nativeBiquadPeakEq(id, freq, gainDb, q, sampleRate)
    fun biquadLowShelf(id: Int, freq: Float, gainDb: Float, q: Float, sampleRate: Int) = nativeBiquadLowShelf(id, freq, gainDb, q, sampleRate)
    fun biquadHighShelf(id: Int, freq: Float, gainDb: Float, q: Float, sampleRate: Int) = nativeBiquadHighShelf(id, freq, gainDb, q, sampleRate)
    fun biquadLowpass(id: Int, freq: Float, q: Float, sampleRate: Int) = nativeBiquadLowpass(id, freq, q, sampleRate)
    fun biquadHighpass(id: Int, freq: Float, q: Float, sampleRate: Int) = nativeBiquadHighpass(id, freq, q, sampleRate)
    fun biquadBandpass(id: Int, freq: Float, q: Float, sampleRate: Int) = nativeBiquadBandpass(id, freq, q, sampleRate)
    fun biquadNotch(id: Int, freq: Float, q: Float, sampleRate: Int) = nativeBiquadNotch(id, freq, q, sampleRate)
    fun biquadProcess(id: Int, left: FloatArray, right: FloatArray) = nativeBiquadProcess(id, left, right)
    fun biquadReset(id: Int) = nativeBiquadReset(id)
    fun biquadResetAll() = nativeBiquadResetAll()

    // LUFS metering
    fun lufsInit(sampleRate: Int) = nativeLufsInit(sampleRate)
    fun lufsProcess(left: FloatArray, right: FloatArray) = nativeLufsProcess(left, right)
    fun lufsGet(): Triple<Float, Float, Float> {
        val r = nativeLufsGet()
        return Triple(r[0], r[1], r[2])
    }
    fun lufsReset() = nativeLufsReset()

    // DC removal
    fun removeDc(left: FloatArray, right: FloatArray) = nativeRemoveDc(left, right)

    // Analysis (moved from Kotlin - now native NEON)
    data class StereoAnalysis(val leftPeak: Float, val rightPeak: Float, val leftRms: Float, val rightRms: Float)
    fun analyzeStereoInterleaved(buffer: FloatArray): StereoAnalysis {
        val r = nativeAnalyzeStereoInterleaved(buffer)
        return StereoAnalysis(r[0], r[1], r[2], r[3])
    }

    data class BlockAnalysis(val energy: Float, val zeroCrossingRate: Float, val peak: Float)
    fun analyzeBlock(samples: FloatArray): BlockAnalysis {
        val r = nativeAnalyzeBlock(samples)
        return BlockAnalysis(r[0], r[1], r[2])
    }

    fun stereoCorrelation(left: FloatArray, right: FloatArray): Float = nativeStereoCorrelation(left, right)

    data class SpectralFeatures(val bass: Float, val mid: Float, val high: Float, val centroid: Float)
    fun spectralFeatures(samples: FloatArray): SpectralFeatures {
        val r = nativeSpectralFeatures(samples)
        return SpectralFeatures(r[0], r[1], r[2], r[3])
    }

    fun spectralSubtract(frame: FloatArray, noiseProfile: FloatArray, reduction: Float) =
        nativeSpectralSubtract(frame, noiseProfile, reduction)

    fun linearToLog(linear: FloatArray, logOut: FloatArray, sampleRate: Int, minFreq: Float = 20f, maxFreq: Float = 20000f) =
        nativeLinearToLog(linear, logOut, sampleRate, minFreq, maxFreq)

    fun applySmoothing(current: FloatArray, target: FloatArray, attack: Float = 0.8f, release: Float = 0.3f) =
        nativeApplySmoothing(current, target, attack, release)

    fun spatialize(left: FloatArray, right: FloatArray, width: Float) = nativeSpatialize(left, right, width)

    fun enhanceContent(samples: FloatArray, contentType: Int) = nativeEnhanceContent(samples, contentType)

    fun updateNoiseFloor(silence: FloatArray, noiseFloor: FloatArray, smoothing: Float = 0.95f) =
        nativeUpdateNoiseFloor(silence, noiseFloor, smoothing)

    // Conversion (NEON optimized)
    fun shortToFloat(src: ShortArray, dst: FloatArray) = nativeShortToFloat(src, dst)
    fun floatToShort(src: FloatArray, dst: ShortArray) = nativeFloatToShort(src, dst)
    fun magnitudeToInterleaved(magnitudes: FloatArray, interleaved: FloatArray) = nativeMagnitudeToInterleaved(magnitudes, interleaved)

    // Additional DSP (NEON optimized)
    fun deesser(samples: FloatArray, thresholdDb: Float = -20f, sampleRate: Int = 48000) =
        nativeDeesser(samples, thresholdDb, sampleRate)
    fun enhanceBass(samples: FloatArray, amount: Float = 0.5f, sampleRate: Int = 48000) =
        nativeEnhanceBass(samples, amount, sampleRate)
    fun exciter(samples: FloatArray, amount: Float = 0.3f) = nativeExciter(samples, amount)
    fun restoreClips(samples: FloatArray) = nativeRestoreClips(samples)
    fun normalizeLoudness(samples: FloatArray, targetLufs: Float = -14f) = nativeNormalizeLoudness(samples, targetLufs)
    fun stereoToMono(left: FloatArray, right: FloatArray, mono: FloatArray) = nativeStereoToMono(left, right, mono)
    fun integratedLoudness(history: FloatArray): Float = nativeIntegratedLoudness(history)
    fun downsampleFft(input: FloatArray, output: FloatArray, smoothingFactor: Float = 0.4f) =
        nativeDownsampleFft(input, output, smoothingFactor)

    // Circular buffer operations (for service visualization)
    fun deinterleaveToCircular(interleaved: FloatArray, leftAcc: FloatArray, rightAcc: FloatArray, position: Int, accSize: Int, frameCount: Int) =
        nativeDeinterleaveToCircular(interleaved, leftAcc, rightAcc, position, accSize, frameCount)
    fun circularToLinear(circular: FloatArray, linear: FloatArray, position: Int, circSize: Int) =
        nativeCircularToLinear(circular, linear, position, circSize)
    fun peakHoldDecay(current: FloatArray, peakHold: FloatArray, decayRate: Float) =
        nativePeakHoldDecay(current, peakHold, decayRate)
    fun averageMagnitudes(left: FloatArray, right: FloatArray, output: FloatArray) =
        nativeAverageMagnitudes(left, right, output)
    fun deinterleaveShortToCircular(interleaved: ShortArray, leftAcc: FloatArray, rightAcc: FloatArray, position: Int, accSize: Int, frameCount: Int) =
        nativeDeinterleaveShortToCircular(interleaved, leftAcc, rightAcc, position, accSize, frameCount)

    // Short buffer analysis - returns [leftPeak, rightPeak, leftSumSq, rightSumSq]
    fun analyzeStereoShort(buffer: ShortArray): LongArray = nativeAnalyzeStereoShort(buffer)

    // Complete processing chain
    fun processChain(
        samples: FloatArray,
        applyEq: Boolean = false,
        eqFilterIds: IntArray? = null,
        applyCompressor: Boolean = false,
        compThresholdDb: Float = -20f,
        compRatio: Float = 4f,
        compAttackMs: Float = 10f,
        compReleaseMs: Float = 100f,
        compMakeupDb: Float = 0f,
        applyLimiter: Boolean = false,
        limitThresholdDb: Float = -0.3f,
        applyStereoWidth: Boolean = false,
        width: Float = 1f,
        applyCrossfeed: Boolean = false,
        crossfeedAmount: Float = 0f,
        sampleRate: Int = 48000
    ) = nativeProcessChain(
        samples,
        applyEq, eqFilterIds, eqFilterIds?.size ?: 0,
        applyCompressor, compThresholdDb, compRatio, compAttackMs, compReleaseMs, compMakeupDb,
        applyLimiter, limitThresholdDb,
        applyStereoWidth, width,
        applyCrossfeed, crossfeedAmount,
        sampleRate
    )

    // Convenience
    fun applyGainDb(samples: FloatArray, gainDb: Float) {
        val linear = Math.pow(10.0, gainDb / 20.0).toFloat()
        nativeApplyGain(samples, linear)
    }

    fun findPeakDb(samples: FloatArray): Float {
        val peak = nativeFindPeak(samples)
        return if (peak > 0.00001f) 20f * kotlin.math.log10(peak) else -100f
    }

    fun calculateRmsDb(samples: FloatArray): Float {
        val rms = nativeCalculateRms(samples)
        return if (rms > 0.00001f) 20f * kotlin.math.log10(rms) else -100f
    }

    fun processStereoDsp(
        interleaved: FloatArray,
        left: FloatArray,
        right: FloatArray,
        process: (FloatArray, FloatArray) -> Unit
    ) {
        nativeStereoDeinterleave(interleaved, left, right)
        process(left, right)
        nativeStereoInterleave(left, right, interleaved)
    }

    fun getHardwareInfo(): String = "S24 Ultra: Cortex-X4 @ 3.3GHz, 4x NEON pipes, 64-byte cache"
}
