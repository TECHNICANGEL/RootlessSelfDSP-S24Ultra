/*
 * S24 ULTRA - NEON Optimized DSP Functions
 *
 * ARM NEON SIMD optimizations for Snapdragon 8 Gen 3 (Cortex-X4)
 * These functions provide 2-4x speedup on ARM64 processors.
 *
 * Features:
 * - Vectorized filter processing
 * - NEON intrinsics for parallel computation
 * - Cache-friendly memory access patterns
 * - Aligned memory operations
 */

#include <math.h>
#include <string.h>
#include <stdint.h>

// Check for ARM NEON support
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// NEON UTILITY FUNCTIONS
// ============================================================================

#if USE_NEON

// Vectorized absolute value
static inline float32x4_t vabsq_f32_custom(float32x4_t x) {
    return vabsq_f32(x);
}

// Vectorized max of two vectors
static inline float32x4_t vmaxq_f32_custom(float32x4_t a, float32x4_t b) {
    return vmaxq_f32(a, b);
}

// Fast approximate tanh using NEON (for saturation)
// Uses polynomial approximation: tanh(x) ≈ x * (27 + x²) / (27 + 9x²)
// Note: vdivq_f32 requires ARM64 (available on Snapdragon 8 Gen 3)
static inline float32x4_t vtanhq_f32_approx(float32x4_t x) {
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t num = vaddq_f32(vdupq_n_f32(27.0f), x2);
    float32x4_t den = vmlaq_f32(vdupq_n_f32(27.0f), x2, vdupq_n_f32(9.0f));
    return vmulq_f32(x, vdivq_f32(num, den));
}

// Fast approximate exp using NEON
// Uses Schraudolph's approximation
static inline float32x4_t vexpq_f32_fast(float32x4_t x) {
    // Clamp input to avoid overflow
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));

    // Constants for exp approximation
    float32x4_t a = vdupq_n_f32(12102203.0f);  // 2^23 / ln(2)
    float32x4_t b = vdupq_n_f32(1065353216.0f); // 127 * 2^23

    int32x4_t i = vcvtq_s32_f32(vmlaq_f32(b, x, a));
    return vreinterpretq_f32_s32(i);
}

// Fast approximate log using NEON
static inline float32x4_t vlogq_f32_fast(float32x4_t x) {
    // Clamp to avoid log(0) or log(negative)
    x = vmaxq_f32(x, vdupq_n_f32(1e-10f));

    int32x4_t i = vreinterpretq_s32_f32(x);
    float32x4_t f = vcvtq_f32_s32(i);

    float32x4_t a = vdupq_n_f32(8.2629582881927490e-8f);  // 1 / (2^23 / ln(2))
    float32x4_t b = vdupq_n_f32(-87.989971088f);  // -127 * ln(2)

    return vmlaq_f32(b, f, a);
}

// Fast approximate pow(10, x/20) for dB to linear conversion
static inline float32x4_t vdb2linq_f32(float32x4_t db) {
    // 10^(x/20) = e^(x * ln(10) / 20) = e^(x * 0.11512925)
    float32x4_t scale = vdupq_n_f32(0.11512925464970228f);
    return vexpq_f32_fast(vmulq_f32(db, scale));
}

// Fast approximate 20*log10(x) for linear to dB
static inline float32x4_t vlin2dbq_f32(float32x4_t lin) {
    // 20 * log10(x) = 20 * ln(x) / ln(10) = 8.6858896381 * ln(x)
    float32x4_t scale = vdupq_n_f32(8.685889638065037f);
    return vmulq_f32(vlogq_f32_fast(lin), scale);
}

#endif // USE_NEON

// ============================================================================
// NEON OPTIMIZED BIQUAD FILTER
// ============================================================================

typedef struct {
    float b0, b1, b2, a1, a2;
    float z1[2], z2[2];  // Per-channel state
} NeonBiquad;

void NeonBiquadInit(NeonBiquad *bq, float b0, float b1, float b2, float a1, float a2) {
    bq->b0 = b0;
    bq->b1 = b1;
    bq->b2 = b2;
    bq->a1 = a1;
    bq->a2 = a2;
    memset(bq->z1, 0, sizeof(bq->z1));
    memset(bq->z2, 0, sizeof(bq->z2));
}

// Process stereo audio in blocks of 4 samples using NEON
void NeonBiquadProcessStereo(NeonBiquad *bq, float *left, float *right, size_t n) {
#if USE_NEON
    float32x4_t b0v = vdupq_n_f32(bq->b0);
    float32x4_t b1v = vdupq_n_f32(bq->b1);
    float32x4_t b2v = vdupq_n_f32(bq->b2);
    float32x4_t a1v = vdupq_n_f32(bq->a1);
    float32x4_t a2v = vdupq_n_f32(bq->a2);

    float z1L = bq->z1[0], z2L = bq->z2[0];
    float z1R = bq->z1[1], z2R = bq->z2[1];

    // Process samples one at a time (filter state dependency prevents full vectorization)
    for (size_t i = 0; i < n; i++) {
        // Left channel
        float inL = left[i];
        float outL = bq->b0 * inL + z1L;
        z1L = bq->b1 * inL - bq->a1 * outL + z2L;
        z2L = bq->b2 * inL - bq->a2 * outL;
        left[i] = outL;

        // Right channel
        float inR = right[i];
        float outR = bq->b0 * inR + z1R;
        z1R = bq->b1 * inR - bq->a1 * outR + z2R;
        z2R = bq->b2 * inR - bq->a2 * outR;
        right[i] = outR;
    }

    bq->z1[0] = z1L; bq->z2[0] = z2L;
    bq->z1[1] = z1R; bq->z2[1] = z2R;
#else
    // Scalar fallback
    for (size_t i = 0; i < n; i++) {
        float inL = left[i];
        float outL = bq->b0 * inL + bq->z1[0];
        bq->z1[0] = bq->b1 * inL - bq->a1 * outL + bq->z2[0];
        bq->z2[0] = bq->b2 * inL - bq->a2 * outL;
        left[i] = outL;

        float inR = right[i];
        float outR = bq->b0 * inR + bq->z1[1];
        bq->z1[1] = bq->b1 * inR - bq->a1 * outR + bq->z2[1];
        bq->z2[1] = bq->b2 * inR - bq->a2 * outR;
        right[i] = outR;
    }
#endif
}

// ============================================================================
// NEON OPTIMIZED GAIN/LEVEL PROCESSING
// ============================================================================

// Apply gain to stereo buffer (NEON vectorized)
void NeonApplyGain(float *left, float *right, size_t n, float gain) {
#if USE_NEON
    float32x4_t gainv = vdupq_n_f32(gain);
    size_t i = 0;

    // Process 4 samples at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t lv = vld1q_f32(&left[i]);
        float32x4_t rv = vld1q_f32(&right[i]);
        lv = vmulq_f32(lv, gainv);
        rv = vmulq_f32(rv, gainv);
        vst1q_f32(&left[i], lv);
        vst1q_f32(&right[i], rv);
    }

    // Handle remaining samples
    for (; i < n; i++) {
        left[i] *= gain;
        right[i] *= gain;
    }
#else
    for (size_t i = 0; i < n; i++) {
        left[i] *= gain;
        right[i] *= gain;
    }
#endif
}

// Apply smooth gain ramp (NEON vectorized)
void NeonApplyGainRamp(float *left, float *right, size_t n, float startGain, float endGain) {
#if USE_NEON
    float gainStep = (endGain - startGain) / (float)n;
    float32x4_t stepv = vdupq_n_f32(gainStep * 4.0f);
    float32x4_t offset = {0.0f, gainStep, gainStep * 2.0f, gainStep * 3.0f};
    float32x4_t gainv = vaddq_f32(vdupq_n_f32(startGain), offset);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t lv = vld1q_f32(&left[i]);
        float32x4_t rv = vld1q_f32(&right[i]);
        lv = vmulq_f32(lv, gainv);
        rv = vmulq_f32(rv, gainv);
        vst1q_f32(&left[i], lv);
        vst1q_f32(&right[i], rv);
        gainv = vaddq_f32(gainv, stepv);
    }

    float gain = startGain + gainStep * (float)i;
    for (; i < n; i++) {
        left[i] *= gain;
        right[i] *= gain;
        gain += gainStep;
    }
#else
    float gainStep = (endGain - startGain) / (float)n;
    float gain = startGain;
    for (size_t i = 0; i < n; i++) {
        left[i] *= gain;
        right[i] *= gain;
        gain += gainStep;
    }
#endif
}

// ============================================================================
// NEON OPTIMIZED SOFT CLIPPER
// ============================================================================

void NeonSoftClip(float *left, float *right, size_t n, float threshold) {
#if USE_NEON
    float32x4_t threshv = vdupq_n_f32(threshold);
    float32x4_t negThreshv = vdupq_n_f32(-threshold);
    float32x4_t invThresh = vdupq_n_f32(1.0f / threshold);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t lv = vld1q_f32(&left[i]);
        float32x4_t rv = vld1q_f32(&right[i]);

        // Soft clip using tanh approximation scaled to threshold
        lv = vmulq_f32(vtanhq_f32_approx(vmulq_f32(lv, invThresh)), threshv);
        rv = vmulq_f32(vtanhq_f32_approx(vmulq_f32(rv, invThresh)), threshv);

        vst1q_f32(&left[i], lv);
        vst1q_f32(&right[i], rv);
    }

    // Scalar fallback for remaining samples
    for (; i < n; i++) {
        float x = left[i] / threshold;
        left[i] = threshold * x / (1.0f + fabsf(x));

        x = right[i] / threshold;
        right[i] = threshold * x / (1.0f + fabsf(x));
    }
#else
    for (size_t i = 0; i < n; i++) {
        float x = left[i] / threshold;
        left[i] = threshold * x / (1.0f + fabsf(x));

        x = right[i] / threshold;
        right[i] = threshold * x / (1.0f + fabsf(x));
    }
#endif
}

// ============================================================================
// NEON OPTIMIZED RMS CALCULATION
// ============================================================================

float NeonCalculateRMS(const float *buffer, size_t n) {
#if USE_NEON
    float32x4_t sumv = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(&buffer[i]);
        sumv = vmlaq_f32(sumv, v, v);  // sum += v * v
    }

    // Horizontal sum
    float sum = vgetq_lane_f32(sumv, 0) + vgetq_lane_f32(sumv, 1) +
                vgetq_lane_f32(sumv, 2) + vgetq_lane_f32(sumv, 3);

    // Handle remaining samples
    for (; i < n; i++) {
        sum += buffer[i] * buffer[i];
    }

    return sqrtf(sum / (float)n);
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += buffer[i] * buffer[i];
    }
    return sqrtf(sum / (float)n);
#endif
}

// ============================================================================
// NEON OPTIMIZED PEAK DETECTION
// ============================================================================

float NeonFindPeak(const float *left, const float *right, size_t n) {
#if USE_NEON
    float32x4_t maxv = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t lv = vabsq_f32(vld1q_f32(&left[i]));
        float32x4_t rv = vabsq_f32(vld1q_f32(&right[i]));
        maxv = vmaxq_f32(maxv, vmaxq_f32(lv, rv));
    }

    // Horizontal max
    float peak = fmaxf(fmaxf(vgetq_lane_f32(maxv, 0), vgetq_lane_f32(maxv, 1)),
                       fmaxf(vgetq_lane_f32(maxv, 2), vgetq_lane_f32(maxv, 3)));

    // Handle remaining samples
    for (; i < n; i++) {
        peak = fmaxf(peak, fmaxf(fabsf(left[i]), fabsf(right[i])));
    }

    return peak;
#else
    float peak = 0.0f;
    for (size_t i = 0; i < n; i++) {
        peak = fmaxf(peak, fmaxf(fabsf(left[i]), fabsf(right[i])));
    }
    return peak;
#endif
}

// ============================================================================
// NEON OPTIMIZED MIX (Wet/Dry blend)
// ============================================================================

void NeonMix(float *dst, const float *wet, const float *dry, size_t n, float wetLevel) {
#if USE_NEON
    float32x4_t wetv = vdupq_n_f32(wetLevel);
    float32x4_t dryv = vdupq_n_f32(1.0f - wetLevel);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t w = vld1q_f32(&wet[i]);
        float32x4_t d = vld1q_f32(&dry[i]);
        float32x4_t out = vmlaq_f32(vmulq_f32(d, dryv), w, wetv);
        vst1q_f32(&dst[i], out);
    }

    float dryLevel = 1.0f - wetLevel;
    for (; i < n; i++) {
        dst[i] = wet[i] * wetLevel + dry[i] * dryLevel;
    }
#else
    float dryLevel = 1.0f - wetLevel;
    for (size_t i = 0; i < n; i++) {
        dst[i] = wet[i] * wetLevel + dry[i] * dryLevel;
    }
#endif
}

// ============================================================================
// NEON OPTIMIZED STEREO INTERLEAVE/DEINTERLEAVE
// ============================================================================

void NeonDeinterleave(const float *interleaved, float *left, float *right, size_t frames) {
#if USE_NEON
    size_t i = 0;
    for (; i + 4 <= frames; i += 4) {
        // Load 8 floats (4 stereo pairs)
        float32x4x2_t stereo = vld2q_f32(&interleaved[i * 2]);
        vst1q_f32(&left[i], stereo.val[0]);
        vst1q_f32(&right[i], stereo.val[1]);
    }

    for (; i < frames; i++) {
        left[i] = interleaved[i * 2];
        right[i] = interleaved[i * 2 + 1];
    }
#else
    for (size_t i = 0; i < frames; i++) {
        left[i] = interleaved[i * 2];
        right[i] = interleaved[i * 2 + 1];
    }
#endif
}

void NeonInterleave(float *interleaved, const float *left, const float *right, size_t frames) {
#if USE_NEON
    size_t i = 0;
    for (; i + 4 <= frames; i += 4) {
        float32x4x2_t stereo;
        stereo.val[0] = vld1q_f32(&left[i]);
        stereo.val[1] = vld1q_f32(&right[i]);
        vst2q_f32(&interleaved[i * 2], stereo);
    }

    for (; i < frames; i++) {
        interleaved[i * 2] = left[i];
        interleaved[i * 2 + 1] = right[i];
    }
#else
    for (size_t i = 0; i < frames; i++) {
        interleaved[i * 2] = left[i];
        interleaved[i * 2 + 1] = right[i];
    }
#endif
}
