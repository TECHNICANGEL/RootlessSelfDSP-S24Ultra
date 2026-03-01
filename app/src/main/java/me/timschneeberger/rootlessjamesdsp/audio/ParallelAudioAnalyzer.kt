package me.timschneeberger.rootlessjamesdsp.audio

import java.util.concurrent.ExecutorService

/**
 * S24 ULTRA - Parallel Audio Analyzer
 * ALL DSP IN C (s24_ultra_dsp.c) - Kotlin is thin wrapper only
 */
object ParallelAudioAnalyzer {

    /**
     * Result container for parallel analysis
     */
    data class AnalysisResult(
        val leftPeak: Float,
        val rightPeak: Float,
        val leftRms: Float,
        val rightRms: Float
    )

    /**
     * Analyze stereo float buffer using native NEON
     * C implementation handles all vectorization
     */
    fun analyzeParallel(
        buffer: FloatArray,
        executor: ExecutorService,  // Kept for API compatibility, not used
        numCores: Int               // Kept for API compatibility, not used
    ): AnalysisResult {
        val result = S24UltraDsp.analyzeStereoInterleaved(buffer)
        return AnalysisResult(result.leftPeak, result.rightPeak, result.leftRms, result.rightRms)
    }

    /**
     * Analyze stereo short buffer - converts to float using native NEON
     */
    fun analyzeParallelShort(
        buffer: ShortArray,
        executor: ExecutorService,
        numCores: Int
    ): AnalysisResult {
        // Convert to float using NEON-optimized native function
        val floatBuffer = FloatArray(buffer.size)
        S24UltraDsp.shortToFloat(buffer, floatBuffer)
        return analyzeParallel(floatBuffer, executor, numCores)
    }
}
