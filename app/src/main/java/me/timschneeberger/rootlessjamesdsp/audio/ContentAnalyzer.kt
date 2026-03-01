package me.timschneeberger.rootlessjamesdsp.audio

import timber.log.Timber
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * S24 ULTRA - Content Analyzer
 *
 * Analyzes audio content asynchronously to detect:
 * - Content type (speech, music, mixed, noise)
 * - Spectral features (bass, mid, high energy)
 * - Loudness (short-term, integrated)
 *
 * Provides recommended DSP parameters based on content.
 * ALL DSP IN C (s24_ultra_dsp.c) - Kotlin manages threading only.
 */
class ContentAnalyzer(private val sampleRate: Int = 48000) {

    companion object {
        private const val TAG = "ContentAnalyzer"
        private const val NUM_BANDS = 32
    }

    private val analyzerExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "Content-Analyzer").apply { priority = Thread.NORM_PRIORITY - 1 }
    }

    private val isRunning = AtomicBoolean(true)

    // Content type detection
    enum class ContentType { UNKNOWN, SPEECH, MUSIC, MIXED, NOISE }
    private val contentType = AtomicReference(ContentType.UNKNOWN)
    private val speechProbability = AtomicReference(0f)
    private val musicProbability = AtomicReference(0f)

    // Recommended parameters
    private val recommendedCompThreshold = AtomicReference(-20f)
    private val recommendedCompRatio = AtomicReference(3f)
    private val recommendedGateThreshold = AtomicReference(-50f)
    private val recommendedBassBoost = AtomicReference(0f)
    private val recommendedTrebleBoost = AtomicReference(0f)
    private val recommendedStereoWidth = AtomicReference(1f)
    private val recommendedLoudnessGain = AtomicReference(0f)

    // Spectral analysis
    private val spectralCentroid = AtomicReference(1000f)
    private val spectralFlux = AtomicReference(0f)
    private val bassEnergy = AtomicReference(0f)
    private val midEnergy = AtomicReference(0f)
    private val highEnergy = AtomicReference(0f)

    // Loudness
    private val shortTermLoudness = AtomicReference(-23f)
    private val integratedLoudness = AtomicReference(-23f)

    // Noise profile
    private val noiseFloor = FloatArray(NUM_BANDS) { -60f }
    @Volatile private var noiseProfileValid = false
    private var consecutiveSilentFrames = 0

    // History for smoothing
    private val contentHistory = Array(10) { ContentType.UNKNOWN }
    private var contentHistoryPos = 0

    fun submitForAnalysis(samples: FloatArray) {
        if (!isRunning.get()) return
        val copy = samples.copyOf()
        analyzerExecutor.submit {
            try { analyzeBlock(copy) }
            catch (e: Exception) { Timber.tag(TAG).w(e, "Analysis failed") }
        }
    }

    fun submitStereoForAnalysis(left: FloatArray, right: FloatArray) {
        if (!isRunning.get()) return
        val copyL = left.copyOf()
        val copyR = right.copyOf()
        analyzerExecutor.submit {
            try { analyzeStereoBlock(copyL, copyR) }
            catch (e: Exception) { Timber.tag(TAG).w(e, "Stereo analysis failed") }
        }
    }

    // Getters
    fun getContentType(): ContentType = contentType.get()
    fun getSpeechProbability(): Float = speechProbability.get()
    fun getMusicProbability(): Float = musicProbability.get()
    fun getRecommendedCompThreshold(): Float = recommendedCompThreshold.get()
    fun getRecommendedCompRatio(): Float = recommendedCompRatio.get()
    fun getRecommendedGateThreshold(): Float = recommendedGateThreshold.get()
    fun getRecommendedBassBoost(): Float = recommendedBassBoost.get()
    fun getRecommendedTrebleBoost(): Float = recommendedTrebleBoost.get()
    fun getRecommendedStereoWidth(): Float = recommendedStereoWidth.get()
    fun getRecommendedLoudnessGain(): Float = recommendedLoudnessGain.get()
    fun getSpectralCentroid(): Float = spectralCentroid.get()
    fun getSpectralFlux(): Float = spectralFlux.get()
    fun getBassEnergy(): Float = bassEnergy.get()
    fun getMidEnergy(): Float = midEnergy.get()
    fun getHighEnergy(): Float = highEnergy.get()
    fun getShortTermLoudness(): Float = shortTermLoudness.get()
    fun getIntegratedLoudness(): Float = integratedLoudness.get()
    fun getNoiseFloor(): FloatArray = noiseFloor.copyOf()
    fun isNoiseProfileValid(): Boolean = noiseProfileValid

    private fun analyzeBlock(samples: FloatArray) {
        // Use native C for all heavy computation
        val analysis = S24UltraDsp.analyzeBlock(samples)
        val energy = analysis.energy
        val zcr = analysis.zeroCrossingRate
        val peak = analysis.peak

        // Detect silence for noise profiling
        if (energy < 0.001f) {
            consecutiveSilentFrames++
            if (consecutiveSilentFrames > 10) {
                S24UltraDsp.updateNoiseFloor(samples, noiseFloor, 0.95f)
                noiseProfileValid = consecutiveSilentFrames > 20
            }
        } else {
            consecutiveSilentFrames = 0
        }

        // Content classification
        val detectedContent = classifyContent(energy, zcr, peak)
        contentHistory[contentHistoryPos] = detectedContent
        contentHistoryPos = (contentHistoryPos + 1) % contentHistory.size

        // Smooth content type (majority voting)
        val contentCounts = contentHistory.groupingBy { it }.eachCount()
        val smoothedContent = contentCounts.maxByOrNull { it.value }?.key ?: ContentType.UNKNOWN
        contentType.set(smoothedContent)

        val total = contentHistory.size.toFloat()
        speechProbability.set((contentCounts[ContentType.SPEECH] ?: 0) / total)
        musicProbability.set((contentCounts[ContentType.MUSIC] ?: 0) / total)

        // Spectral features via native
        val spectral = S24UltraDsp.spectralFeatures(samples)
        bassEnergy.set(spectral.bass)
        midEnergy.set(spectral.mid)
        highEnergy.set(spectral.high)
        spectralCentroid.set(spectral.centroid)

        // Update recommendations
        updateRecommendations(smoothedContent, energy, peak)

        // Loudness
        val loudnessDb = if (energy > 0.00001f) 20f * kotlin.math.log10(energy) else -70f
        shortTermLoudness.set(loudnessDb)
        integratedLoudness.set(integratedLoudness.get() * 0.99f + loudnessDb * 0.01f)
    }

    private fun analyzeStereoBlock(left: FloatArray, right: FloatArray) {
        // Mono sum for content analysis (NEON optimized)
        val mono = FloatArray(left.size)
        S24UltraDsp.stereoToMono(left, right, mono)
        analyzeBlock(mono)

        // Stereo correlation via native
        val correlation = S24UltraDsp.stereoCorrelation(left, right)
        val recWidth = when {
            correlation > 0.8f -> 1.3f
            correlation > 0.5f -> 1.1f
            else -> 1.0f
        }
        recommendedStereoWidth.set(recWidth)
    }

    private fun classifyContent(energy: Float, zcr: Float, peak: Float): ContentType {
        val crest = if (energy > 0.00001f) peak / energy else 0f
        return when {
            energy < 0.0001f -> ContentType.NOISE
            zcr > 0.25f && energy < 0.05f -> ContentType.SPEECH
            zcr < 0.1f && crest < 5f -> ContentType.MUSIC
            zcr in 0.1f..0.25f -> ContentType.MIXED
            else -> ContentType.UNKNOWN
        }
    }

    private fun updateRecommendations(content: ContentType, energy: Float, peak: Float) {
        when (content) {
            ContentType.SPEECH -> {
                recommendedCompThreshold.set(-18f); recommendedCompRatio.set(2.5f)
                recommendedGateThreshold.set(-45f); recommendedBassBoost.set(-1f)
                recommendedTrebleBoost.set(2f)
            }
            ContentType.MUSIC -> {
                recommendedCompThreshold.set(-24f); recommendedCompRatio.set(2f)
                recommendedGateThreshold.set(-60f); recommendedBassBoost.set(1f)
                recommendedTrebleBoost.set(0.5f)
            }
            ContentType.MIXED -> {
                recommendedCompThreshold.set(-20f); recommendedCompRatio.set(3f)
                recommendedGateThreshold.set(-50f); recommendedBassBoost.set(0f)
                recommendedTrebleBoost.set(1f)
            }
            ContentType.NOISE -> {
                recommendedCompThreshold.set(-30f); recommendedCompRatio.set(1.5f)
                recommendedGateThreshold.set(-35f); recommendedBassBoost.set(-2f)
                recommendedTrebleBoost.set(-1f)
            }
            else -> {
                recommendedCompThreshold.set(-20f); recommendedCompRatio.set(3f)
                recommendedGateThreshold.set(-50f); recommendedBassBoost.set(0f)
                recommendedTrebleBoost.set(0f)
            }
        }
        val targetLufs = -14f
        val gainNeeded = (targetLufs - integratedLoudness.get()).coerceIn(-12f, 12f)
        recommendedLoudnessGain.set(gainNeeded)
    }

    fun shutdown() {
        isRunning.set(false)
        analyzerExecutor.shutdown()
        // PERFORMANCE: Proper shutdown with await to prevent resource leaks
        try {
            if (!analyzerExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                analyzerExecutor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            analyzerExecutor.shutdownNow()
            Thread.currentThread().interrupt()
        }
    }
}
