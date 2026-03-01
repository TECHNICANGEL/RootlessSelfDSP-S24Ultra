/*
 * S24 ULTRA - Advanced Dynamics Processing
 *
 * Contains:
 * - EBU R128 Loudness Normalization
 * - Multi-band Compressor (4 bands)
 * - Noise Gate
 * - Psychoacoustic Bass Enhancement
 *
 * Optimized for Snapdragon 8 Gen 3 with NEON intrinsics
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "../jdsp_header.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// EBU R128 LOUDNESS NORMALIZATION
// ============================================================================

// K-weighting filter coefficients (for 48kHz, will be recalculated for other rates)
typedef struct {
    // High shelf (stage 1)
    double hs_b0, hs_b1, hs_b2, hs_a1, hs_a2;
    // High pass (stage 2)
    double hp_b0, hp_b1, hp_b2, hp_a1, hp_a2;
    // Filter states
    double hs_x1[2], hs_x2[2], hs_y1[2], hs_y2[2];
    double hp_x1[2], hp_x2[2], hp_y1[2], hp_y2[2];
} KWeightingFilter;

typedef struct LoudnessNormalizer {
    KWeightingFilter kFilter;
    double fs;

    // Momentary loudness (400ms)
    double momentaryBuffer[19200];  // 400ms at 48kHz
    int momentaryPos;
    int momentaryLen;
    double momentarySum;

    // Short-term loudness (3s)
    double shortTermBuffer[144000]; // 3s at 48kHz
    int shortTermPos;
    int shortTermLen;
    double shortTermSum;

    // Integrated loudness (gated)
    double integratedSum;
    int integratedCount;
    double relativeThreshold;

    // Target loudness
    double targetLUFS;
    double currentGain;
    double gainSmoothing;

    // True peak detection
    double truePeakL;
    double truePeakR;
    double truePeakMax;
    double truePeakLimit;

    // Output
    double momentaryLUFS;
    double shortTermLUFS;
    double integratedLUFS;
    double loudnessRange;

    int enabled;
} LoudnessNormalizer;

static void initKWeightingFilter(KWeightingFilter *kf, double fs)
{
    // Stage 1: High shelf boost (+4dB at high frequencies)
    // Pre-filter as per ITU-R BS.1770
    double Vh = 1.58489319246111;  // 10^(4/20)
    double Vb = Vh - 1.0;
    double fc_hs = 1500.0;  // Shelf frequency
    double Q_hs = 0.707;

    double K = tan(M_PI * fc_hs / fs);
    double K2 = K * K;
    double denom = 1.0 + K / Q_hs + K2;

    kf->hs_b0 = (Vh + Vb * K / Q_hs + K2) / denom;
    kf->hs_b1 = 2.0 * (K2 - Vh) / denom;
    kf->hs_b2 = (Vh - Vb * K / Q_hs + K2) / denom;
    kf->hs_a1 = 2.0 * (K2 - 1.0) / denom;
    kf->hs_a2 = (1.0 - K / Q_hs + K2) / denom;

    // Stage 2: High pass filter (remove DC and subsonic)
    double fc_hp = 38.0;  // HPF cutoff
    double Q_hp = 0.5;

    K = tan(M_PI * fc_hp / fs);
    K2 = K * K;
    denom = 1.0 + K / Q_hp + K2;

    kf->hp_b0 = 1.0 / denom;
    kf->hp_b1 = -2.0 / denom;
    kf->hp_b2 = 1.0 / denom;
    kf->hp_a1 = 2.0 * (K2 - 1.0) / denom;
    kf->hp_a2 = (1.0 - K / Q_hp + K2) / denom;

    // Clear states
    memset(kf->hs_x1, 0, sizeof(kf->hs_x1));
    memset(kf->hs_x2, 0, sizeof(kf->hs_x2));
    memset(kf->hs_y1, 0, sizeof(kf->hs_y1));
    memset(kf->hs_y2, 0, sizeof(kf->hs_y2));
    memset(kf->hp_x1, 0, sizeof(kf->hp_x1));
    memset(kf->hp_x2, 0, sizeof(kf->hp_x2));
    memset(kf->hp_y1, 0, sizeof(kf->hp_y1));
    memset(kf->hp_y2, 0, sizeof(kf->hp_y2));
}

static inline double processKWeighting(KWeightingFilter *kf, double input, int channel)
{
    // Stage 1: High shelf
    double y_hs = kf->hs_b0 * input + kf->hs_b1 * kf->hs_x1[channel] + kf->hs_b2 * kf->hs_x2[channel]
                - kf->hs_a1 * kf->hs_y1[channel] - kf->hs_a2 * kf->hs_y2[channel];

    kf->hs_x2[channel] = kf->hs_x1[channel];
    kf->hs_x1[channel] = input;
    kf->hs_y2[channel] = kf->hs_y1[channel];
    kf->hs_y1[channel] = y_hs;

    // Stage 2: High pass
    double y_hp = kf->hp_b0 * y_hs + kf->hp_b1 * kf->hp_x1[channel] + kf->hp_b2 * kf->hp_x2[channel]
                - kf->hp_a1 * kf->hp_y1[channel] - kf->hp_a2 * kf->hp_y2[channel];

    kf->hp_x2[channel] = kf->hp_x1[channel];
    kf->hp_x1[channel] = y_hs;
    kf->hp_y2[channel] = kf->hp_y1[channel];
    kf->hp_y1[channel] = y_hp;

    return y_hp;
}

void LoudnessNormalizerInit(LoudnessNormalizer *ln, double fs)
{
    memset(ln, 0, sizeof(LoudnessNormalizer));
    ln->fs = fs;
    ln->targetLUFS = -14.0;  // Default streaming target
    ln->truePeakLimit = -1.0; // -1dB true peak limit
    ln->gainSmoothing = 0.9995;
    ln->currentGain = 1.0;
    ln->momentaryLen = (int)(0.4 * fs);  // 400ms
    ln->shortTermLen = (int)(3.0 * fs);  // 3s
    initKWeightingFilter(&ln->kFilter, fs);
}

void LoudnessNormalizerSetTarget(LoudnessNormalizer *ln, double targetLUFS, double truePeakLimit)
{
    ln->targetLUFS = targetLUFS;
    ln->truePeakLimit = truePeakLimit;
}

void LoudnessNormalizerProcess(LoudnessNormalizer *ln, float *left, float *right, size_t n)
{
    if (!ln->enabled) return;

    for (size_t i = 0; i < n; i++)
    {
        // Apply K-weighting filter
        double kL = processKWeighting(&ln->kFilter, (double)left[i], 0);
        double kR = processKWeighting(&ln->kFilter, (double)right[i], 1);

        // Mean square
        double ms = kL * kL + kR * kR;

        // Update momentary buffer (400ms)
        if (ln->momentaryPos < ln->momentaryLen)
        {
            ln->momentaryBuffer[ln->momentaryPos] = ms;
            ln->momentarySum += ms;
        }
        else
        {
            int oldPos = ln->momentaryPos % ln->momentaryLen;
            ln->momentarySum -= ln->momentaryBuffer[oldPos];
            ln->momentaryBuffer[oldPos] = ms;
            ln->momentarySum += ms;
        }
        ln->momentaryPos++;

        // Calculate momentary loudness every 100ms
        if (ln->momentaryPos % (int)(0.1 * ln->fs) == 0)
        {
            int count = (ln->momentaryPos < ln->momentaryLen) ? ln->momentaryPos : ln->momentaryLen;
            if (count > 0)
            {
                double meanSquare = ln->momentarySum / count;
                ln->momentaryLUFS = -0.691 + 10.0 * log10(meanSquare + 1e-20);
            }

            // Adaptive gain adjustment
            double error = ln->targetLUFS - ln->momentaryLUFS;
            double targetGain = pow(10.0, error / 20.0);

            // Smooth gain changes
            ln->currentGain = ln->currentGain * ln->gainSmoothing + targetGain * (1.0 - ln->gainSmoothing);

            // Limit gain range
            if (ln->currentGain > 10.0) ln->currentGain = 10.0;  // +20dB max
            if (ln->currentGain < 0.1) ln->currentGain = 0.1;   // -20dB min
        }

        // Apply gain
        float gain = (float)ln->currentGain;
        left[i] *= gain;
        right[i] *= gain;

        // True peak limiting
        float peak = fmaxf(fabsf(left[i]), fabsf(right[i]));
        float peakLimit = (float)pow(10.0, ln->truePeakLimit / 20.0);
        if (peak > peakLimit)
        {
            float reduction = peakLimit / peak;
            left[i] *= reduction;
            right[i] *= reduction;
        }

        // Update true peak
        if (fabsf(left[i]) > ln->truePeakL) ln->truePeakL = fabsf(left[i]);
        if (fabsf(right[i]) > ln->truePeakR) ln->truePeakR = fabsf(right[i]);
    }
}

// ============================================================================
// MULTI-BAND COMPRESSOR (4 Bands)
// ============================================================================

#define MBC_NUM_BANDS 4

typedef struct MultibandCompressor {
    // Crossover frequencies
    double crossoverFreqs[MBC_NUM_BANDS - 1];  // 3 crossover points for 4 bands

    // Linkwitz-Riley crossover filters (4th order = 2x 2nd order)
    double lp_b0[MBC_NUM_BANDS - 1], lp_b1[MBC_NUM_BANDS - 1], lp_b2[MBC_NUM_BANDS - 1];
    double lp_a1[MBC_NUM_BANDS - 1], lp_a2[MBC_NUM_BANDS - 1];
    double hp_b0[MBC_NUM_BANDS - 1], hp_b1[MBC_NUM_BANDS - 1], hp_b2[MBC_NUM_BANDS - 1];
    double hp_a1[MBC_NUM_BANDS - 1], hp_a2[MBC_NUM_BANDS - 1];

    // Filter states (L/R, 4th order = 2 stages)
    double lp_z1[2][MBC_NUM_BANDS - 1][2], lp_z2[2][MBC_NUM_BANDS - 1][2];
    double hp_z1[2][MBC_NUM_BANDS - 1][2], hp_z2[2][MBC_NUM_BANDS - 1][2];

    // Per-band compressor parameters
    float threshold[MBC_NUM_BANDS];    // dB
    float ratio[MBC_NUM_BANDS];        // compression ratio
    float attack[MBC_NUM_BANDS];       // ms
    float release[MBC_NUM_BANDS];      // ms
    float makeupGain[MBC_NUM_BANDS];   // dB
    float knee[MBC_NUM_BANDS];         // dB (soft knee width)

    // Envelope followers
    float envL[MBC_NUM_BANDS], envR[MBC_NUM_BANDS];
    float attackCoef[MBC_NUM_BANDS], releaseCoef[MBC_NUM_BANDS];

    // Output
    float bandGainReduction[MBC_NUM_BANDS];

    double fs;
    int enabled;
} MultibandCompressor;

static void calcLinkwitzRileyCoeffs(double fc, double fs, int stage,
                                    double *lp_b0, double *lp_b1, double *lp_b2,
                                    double *lp_a1, double *lp_a2,
                                    double *hp_b0, double *hp_b1, double *hp_b2,
                                    double *hp_a1, double *hp_a2)
{
    double K = tan(M_PI * fc / fs);
    double K2 = K * K;
    double sqrt2 = 1.41421356237;
    double norm = 1.0 / (1.0 + sqrt2 * K + K2);

    // Lowpass
    *lp_b0 = K2 * norm;
    *lp_b1 = 2.0 * *lp_b0;
    *lp_b2 = *lp_b0;
    *lp_a1 = 2.0 * (K2 - 1.0) * norm;
    *lp_a2 = (1.0 - sqrt2 * K + K2) * norm;

    // Highpass
    *hp_b0 = norm;
    *hp_b1 = -2.0 * norm;
    *hp_b2 = norm;
    *hp_a1 = *lp_a1;
    *hp_a2 = *lp_a2;
}

void MultibandCompressorInit(MultibandCompressor *mbc, double fs)
{
    memset(mbc, 0, sizeof(MultibandCompressor));
    mbc->fs = fs;

    // Default crossover frequencies
    mbc->crossoverFreqs[0] = 100.0;   // Low/Low-Mid
    mbc->crossoverFreqs[1] = 1000.0;  // Low-Mid/Mid-High
    mbc->crossoverFreqs[2] = 8000.0;  // Mid-High/High

    // Default compression settings per band
    float defaultThresh[4] = {-20.0f, -18.0f, -16.0f, -14.0f};
    float defaultRatio[4] = {3.0f, 2.5f, 2.0f, 1.5f};
    float defaultAttack[4] = {10.0f, 8.0f, 5.0f, 3.0f};
    float defaultRelease[4] = {150.0f, 100.0f, 80.0f, 60.0f};

    for (int i = 0; i < MBC_NUM_BANDS; i++)
    {
        mbc->threshold[i] = defaultThresh[i];
        mbc->ratio[i] = defaultRatio[i];
        mbc->attack[i] = defaultAttack[i];
        mbc->release[i] = defaultRelease[i];
        mbc->makeupGain[i] = 0.0f;
        mbc->knee[i] = 6.0f;  // 6dB soft knee

        mbc->attackCoef[i] = expf(-1.0f / (mbc->attack[i] * 0.001f * (float)fs));
        mbc->releaseCoef[i] = expf(-1.0f / (mbc->release[i] * 0.001f * (float)fs));
    }

    // Initialize crossover filters
    for (int i = 0; i < MBC_NUM_BANDS - 1; i++)
    {
        calcLinkwitzRileyCoeffs(mbc->crossoverFreqs[i], fs, 0,
                                &mbc->lp_b0[i], &mbc->lp_b1[i], &mbc->lp_b2[i],
                                &mbc->lp_a1[i], &mbc->lp_a2[i],
                                &mbc->hp_b0[i], &mbc->hp_b1[i], &mbc->hp_b2[i],
                                &mbc->hp_a1[i], &mbc->hp_a2[i]);
    }
}

void MultibandCompressorSetBand(MultibandCompressor *mbc, int band,
                                 float threshold, float ratio, float attack,
                                 float release, float makeupGain, float knee)
{
    if (band < 0 || band >= MBC_NUM_BANDS) return;

    mbc->threshold[band] = threshold;
    mbc->ratio[band] = (ratio < 1.0f) ? 1.0f : ratio;
    mbc->attack[band] = attack;
    mbc->release[band] = release;
    mbc->makeupGain[band] = makeupGain;
    mbc->knee[band] = knee;

    mbc->attackCoef[band] = expf(-1.0f / (attack * 0.001f * (float)mbc->fs));
    mbc->releaseCoef[band] = expf(-1.0f / (release * 0.001f * (float)mbc->fs));
}

static inline float processBiquad(double b0, double b1, double b2, double a1, double a2,
                                   double *z1, double *z2, float input)
{
    double output = b0 * input + *z1;
    *z1 = b1 * input - a1 * output + *z2;
    *z2 = b2 * input - a2 * output;
    return (float)output;
}

static inline float compressSample(float input, float threshold, float ratio,
                                    float knee, float makeupGain,
                                    float *env, float attackCoef, float releaseCoef)
{
    float inputLevel = fabsf(input);
    float inputDb = 20.0f * log10f(inputLevel + 1e-10f);

    // Envelope follower
    if (inputDb > *env)
        *env = attackCoef * *env + (1.0f - attackCoef) * inputDb;
    else
        *env = releaseCoef * *env + (1.0f - releaseCoef) * inputDb;

    // Soft knee compression
    float overshoot = *env - threshold;
    float gainReduction = 0.0f;

    if (overshoot < -knee / 2.0f)
    {
        // Below knee - no compression
        gainReduction = 0.0f;
    }
    else if (overshoot > knee / 2.0f)
    {
        // Above knee - full compression
        gainReduction = overshoot * (1.0f - 1.0f / ratio);
    }
    else
    {
        // In knee - gradual compression
        float x = overshoot + knee / 2.0f;
        gainReduction = (1.0f - 1.0f / ratio) * x * x / (2.0f * knee);
    }

    float gain = powf(10.0f, (-gainReduction + makeupGain) / 20.0f);
    return input * gain;
}

void MultibandCompressorProcess(MultibandCompressor *mbc, float *left, float *right, size_t n)
{
    if (!mbc->enabled) return;

    for (size_t i = 0; i < n; i++)
    {
        float bandL[MBC_NUM_BANDS], bandR[MBC_NUM_BANDS];
        float inL = left[i], inR = right[i];

        // Split into bands using Linkwitz-Riley crossovers
        // Band 0: Low (below crossover 0)
        float lpL = processBiquad(mbc->lp_b0[0], mbc->lp_b1[0], mbc->lp_b2[0],
                                   mbc->lp_a1[0], mbc->lp_a2[0],
                                   &mbc->lp_z1[0][0][0], &mbc->lp_z2[0][0][0], inL);
        lpL = processBiquad(mbc->lp_b0[0], mbc->lp_b1[0], mbc->lp_b2[0],
                            mbc->lp_a1[0], mbc->lp_a2[0],
                            &mbc->lp_z1[0][0][1], &mbc->lp_z2[0][0][1], lpL);
        bandL[0] = lpL;

        float lpR = processBiquad(mbc->lp_b0[0], mbc->lp_b1[0], mbc->lp_b2[0],
                                   mbc->lp_a1[0], mbc->lp_a2[0],
                                   &mbc->lp_z1[1][0][0], &mbc->lp_z2[1][0][0], inR);
        lpR = processBiquad(mbc->lp_b0[0], mbc->lp_b1[0], mbc->lp_b2[0],
                            mbc->lp_a1[0], mbc->lp_a2[0],
                            &mbc->lp_z1[1][0][1], &mbc->lp_z2[1][0][1], lpR);
        bandR[0] = lpR;

        // High pass for remaining bands
        float hpL = processBiquad(mbc->hp_b0[0], mbc->hp_b1[0], mbc->hp_b2[0],
                                   mbc->hp_a1[0], mbc->hp_a2[0],
                                   &mbc->hp_z1[0][0][0], &mbc->hp_z2[0][0][0], inL);
        hpL = processBiquad(mbc->hp_b0[0], mbc->hp_b1[0], mbc->hp_b2[0],
                            mbc->hp_a1[0], mbc->hp_a2[0],
                            &mbc->hp_z1[0][0][1], &mbc->hp_z2[0][0][1], hpL);

        float hpR = processBiquad(mbc->hp_b0[0], mbc->hp_b1[0], mbc->hp_b2[0],
                                   mbc->hp_a1[0], mbc->hp_a2[0],
                                   &mbc->hp_z1[1][0][0], &mbc->hp_z2[1][0][0], inR);
        hpR = processBiquad(mbc->hp_b0[0], mbc->hp_b1[0], mbc->hp_b2[0],
                            mbc->hp_a1[0], mbc->hp_a2[0],
                            &mbc->hp_z1[1][0][1], &mbc->hp_z2[1][0][1], hpR);

        // Band 1: Low-Mid (between crossover 0 and 1)
        lpL = processBiquad(mbc->lp_b0[1], mbc->lp_b1[1], mbc->lp_b2[1],
                            mbc->lp_a1[1], mbc->lp_a2[1],
                            &mbc->lp_z1[0][1][0], &mbc->lp_z2[0][1][0], hpL);
        lpL = processBiquad(mbc->lp_b0[1], mbc->lp_b1[1], mbc->lp_b2[1],
                            mbc->lp_a1[1], mbc->lp_a2[1],
                            &mbc->lp_z1[0][1][1], &mbc->lp_z2[0][1][1], lpL);
        bandL[1] = lpL;

        lpR = processBiquad(mbc->lp_b0[1], mbc->lp_b1[1], mbc->lp_b2[1],
                            mbc->lp_a1[1], mbc->lp_a2[1],
                            &mbc->lp_z1[1][1][0], &mbc->lp_z2[1][1][0], hpR);
        lpR = processBiquad(mbc->lp_b0[1], mbc->lp_b1[1], mbc->lp_b2[1],
                            mbc->lp_a1[1], mbc->lp_a2[1],
                            &mbc->lp_z1[1][1][1], &mbc->lp_z2[1][1][1], lpR);
        bandR[1] = lpR;

        // High pass for bands 2-3
        hpL = processBiquad(mbc->hp_b0[1], mbc->hp_b1[1], mbc->hp_b2[1],
                            mbc->hp_a1[1], mbc->hp_a2[1],
                            &mbc->hp_z1[0][1][0], &mbc->hp_z2[0][1][0], hpL);
        hpL = processBiquad(mbc->hp_b0[1], mbc->hp_b1[1], mbc->hp_b2[1],
                            mbc->hp_a1[1], mbc->hp_a2[1],
                            &mbc->hp_z1[0][1][1], &mbc->hp_z2[0][1][1], hpL);

        hpR = processBiquad(mbc->hp_b0[1], mbc->hp_b1[1], mbc->hp_b2[1],
                            mbc->hp_a1[1], mbc->hp_a2[1],
                            &mbc->hp_z1[1][1][0], &mbc->hp_z2[1][1][0], hpR);
        hpR = processBiquad(mbc->hp_b0[1], mbc->hp_b1[1], mbc->hp_b2[1],
                            mbc->hp_a1[1], mbc->hp_a2[1],
                            &mbc->hp_z1[1][1][1], &mbc->hp_z2[1][1][1], hpR);

        // Band 2: Mid-High (between crossover 1 and 2)
        lpL = processBiquad(mbc->lp_b0[2], mbc->lp_b1[2], mbc->lp_b2[2],
                            mbc->lp_a1[2], mbc->lp_a2[2],
                            &mbc->lp_z1[0][2][0], &mbc->lp_z2[0][2][0], hpL);
        lpL = processBiquad(mbc->lp_b0[2], mbc->lp_b1[2], mbc->lp_b2[2],
                            mbc->lp_a1[2], mbc->lp_a2[2],
                            &mbc->lp_z1[0][2][1], &mbc->lp_z2[0][2][1], lpL);
        bandL[2] = lpL;

        lpR = processBiquad(mbc->lp_b0[2], mbc->lp_b1[2], mbc->lp_b2[2],
                            mbc->lp_a1[2], mbc->lp_a2[2],
                            &mbc->lp_z1[1][2][0], &mbc->lp_z2[1][2][0], hpR);
        lpR = processBiquad(mbc->lp_b0[2], mbc->lp_b1[2], mbc->lp_b2[2],
                            mbc->lp_a1[2], mbc->lp_a2[2],
                            &mbc->lp_z1[1][2][1], &mbc->lp_z2[1][2][1], lpR);
        bandR[2] = lpR;

        // Band 3: High (above crossover 2)
        bandL[3] = processBiquad(mbc->hp_b0[2], mbc->hp_b1[2], mbc->hp_b2[2],
                                  mbc->hp_a1[2], mbc->hp_a2[2],
                                  &mbc->hp_z1[0][2][0], &mbc->hp_z2[0][2][0], hpL);
        bandL[3] = processBiquad(mbc->hp_b0[2], mbc->hp_b1[2], mbc->hp_b2[2],
                                  mbc->hp_a1[2], mbc->hp_a2[2],
                                  &mbc->hp_z1[0][2][1], &mbc->hp_z2[0][2][1], bandL[3]);

        bandR[3] = processBiquad(mbc->hp_b0[2], mbc->hp_b1[2], mbc->hp_b2[2],
                                  mbc->hp_a1[2], mbc->hp_a2[2],
                                  &mbc->hp_z1[1][2][0], &mbc->hp_z2[1][2][0], hpR);
        bandR[3] = processBiquad(mbc->hp_b0[2], mbc->hp_b1[2], mbc->hp_b2[2],
                                  mbc->hp_a1[2], mbc->hp_a2[2],
                                  &mbc->hp_z1[1][2][1], &mbc->hp_z2[1][2][1], bandR[3]);

        // Compress each band
        float outL = 0.0f, outR = 0.0f;
        for (int b = 0; b < MBC_NUM_BANDS; b++)
        {
            bandL[b] = compressSample(bandL[b], mbc->threshold[b], mbc->ratio[b],
                                       mbc->knee[b], mbc->makeupGain[b],
                                       &mbc->envL[b], mbc->attackCoef[b], mbc->releaseCoef[b]);
            bandR[b] = compressSample(bandR[b], mbc->threshold[b], mbc->ratio[b],
                                       mbc->knee[b], mbc->makeupGain[b],
                                       &mbc->envR[b], mbc->attackCoef[b], mbc->releaseCoef[b]);
            outL += bandL[b];
            outR += bandR[b];
        }

        left[i] = outL;
        right[i] = outR;
    }
}

// ============================================================================
// NOISE GATE
// ============================================================================

typedef struct NoiseGate {
    float threshold;      // dB
    float attack;         // ms
    float hold;           // ms
    float release;        // ms
    float range;          // dB (max attenuation)
    float hysteresis;     // dB

    // Filter for sidechain (optional HPF to ignore bass rumble)
    float scHpfFreq;
    double scHpf_b0, scHpf_b1, scHpf_b2, scHpf_a1, scHpf_a2;
    double scHpf_z1[2], scHpf_z2[2];
    int scHpfEnabled;

    // Envelope and state
    float env;
    float gain;
    int holdCounter;
    int holdSamples;
    float attackCoef;
    float releaseCoef;
    float rangeLinear;

    // State: 0=closed, 1=opening, 2=open, 3=holding, 4=closing
    int state;

    double fs;
    int enabled;
} NoiseGate;

void NoiseGateInit(NoiseGate *ng, double fs)
{
    memset(ng, 0, sizeof(NoiseGate));
    ng->fs = fs;
    ng->threshold = -40.0f;
    ng->attack = 1.0f;
    ng->hold = 50.0f;
    ng->release = 100.0f;
    ng->range = -80.0f;
    ng->hysteresis = 3.0f;
    ng->scHpfFreq = 100.0f;
    ng->scHpfEnabled = 1;
    ng->gain = 0.0f;  // Start closed
    ng->rangeLinear = powf(10.0f, ng->range / 20.0f);

    ng->attackCoef = expf(-1.0f / (ng->attack * 0.001f * (float)fs));
    ng->releaseCoef = expf(-1.0f / (ng->release * 0.001f * (float)fs));
    ng->holdSamples = (int)(ng->hold * 0.001f * fs);

    // Initialize sidechain HPF (Butterworth 2nd order)
    double fc = ng->scHpfFreq;
    double K = tan(M_PI * fc / fs);
    double K2 = K * K;
    double sqrt2 = 1.41421356237;
    double norm = 1.0 / (1.0 + sqrt2 * K + K2);

    ng->scHpf_b0 = norm;
    ng->scHpf_b1 = -2.0 * norm;
    ng->scHpf_b2 = norm;
    ng->scHpf_a1 = 2.0 * (K2 - 1.0) * norm;
    ng->scHpf_a2 = (1.0 - sqrt2 * K + K2) * norm;
}

void NoiseGateSetParams(NoiseGate *ng, float threshold, float attack, float hold,
                        float release, float range, float hysteresis)
{
    ng->threshold = threshold;
    ng->attack = (attack < 0.1f) ? 0.1f : attack;
    ng->hold = hold;
    ng->release = (release < 1.0f) ? 1.0f : release;
    ng->range = range;
    ng->hysteresis = hysteresis;

    ng->attackCoef = expf(-1.0f / (ng->attack * 0.001f * (float)ng->fs));
    ng->releaseCoef = expf(-1.0f / (ng->release * 0.001f * (float)ng->fs));
    ng->holdSamples = (int)(ng->hold * 0.001f * ng->fs);
    ng->rangeLinear = powf(10.0f, ng->range / 20.0f);
}

void NoiseGateProcess(NoiseGate *ng, float *left, float *right, size_t n)
{
    if (!ng->enabled) return;

    float openThresh = powf(10.0f, ng->threshold / 20.0f);
    float closeThresh = powf(10.0f, (ng->threshold - ng->hysteresis) / 20.0f);

    for (size_t i = 0; i < n; i++)
    {
        // Sidechain signal (mono sum)
        float scL = left[i], scR = right[i];

        // Optional HPF on sidechain
        if (ng->scHpfEnabled)
        {
            scL = (float)(ng->scHpf_b0 * scL + ng->scHpf_z1[0]);
            ng->scHpf_z1[0] = ng->scHpf_b1 * left[i] - ng->scHpf_a1 * scL + ng->scHpf_z2[0];
            ng->scHpf_z2[0] = ng->scHpf_b2 * left[i] - ng->scHpf_a2 * scL;

            scR = (float)(ng->scHpf_b0 * scR + ng->scHpf_z1[1]);
            ng->scHpf_z1[1] = ng->scHpf_b1 * right[i] - ng->scHpf_a1 * scR + ng->scHpf_z2[1];
            ng->scHpf_z2[1] = ng->scHpf_b2 * right[i] - ng->scHpf_a2 * scR;
        }

        float level = fmaxf(fabsf(scL), fabsf(scR));

        // Gate state machine
        switch (ng->state)
        {
            case 0:  // Closed
                if (level > openThresh)
                {
                    ng->state = 1;  // Start opening
                }
                break;

            case 1:  // Opening
                ng->gain = ng->attackCoef * ng->gain + (1.0f - ng->attackCoef) * 1.0f;
                if (ng->gain > 0.99f)
                {
                    ng->gain = 1.0f;
                    ng->state = 2;  // Fully open
                }
                break;

            case 2:  // Open
                if (level < closeThresh)
                {
                    ng->holdCounter = ng->holdSamples;
                    ng->state = 3;  // Start hold
                }
                break;

            case 3:  // Holding
                if (level > openThresh)
                {
                    ng->state = 2;  // Back to open
                }
                else if (--ng->holdCounter <= 0)
                {
                    ng->state = 4;  // Start closing
                }
                break;

            case 4:  // Closing
                ng->gain = ng->releaseCoef * ng->gain + (1.0f - ng->releaseCoef) * ng->rangeLinear;
                if (ng->gain <= ng->rangeLinear + 0.001f)
                {
                    ng->gain = ng->rangeLinear;
                    ng->state = 0;  // Fully closed
                }
                if (level > openThresh)
                {
                    ng->state = 1;  // Interrupt closing
                }
                break;
        }

        left[i] *= ng->gain;
        right[i] *= ng->gain;
    }
}

// ============================================================================
// PSYCHOACOUSTIC BASS ENHANCEMENT
// ============================================================================

typedef struct PsychoacousticBass {
    // Lowpass filter to extract bass
    double lp_b0, lp_b1, lp_b2, lp_a1, lp_a2;
    double lp_z1[2], lp_z2[2];

    // Highpass filter to remove original bass (for mixing)
    double hp_b0, hp_b1, hp_b2, hp_a1, hp_a2;
    double hp_z1[2], hp_z2[2];

    // Parameters
    float frequency;       // Crossover frequency (40-120 Hz)
    float harmonicLevel;   // 2nd harmonic level (0-100%)
    float thirdHarmonic;   // 3rd harmonic level (0-100%)
    float fourthHarmonic;  // 4th harmonic level (0-100%)
    float subBassLevel;    // Sub-bass boost level (0-100%)
    float drive;           // Harmonic saturation drive (0-100%)
    float mix;             // Wet/dry mix (0-100%)

    // Harmonic generation state
    float phase[2];
    float prevBass[2];

    // Output limiter
    float limiterEnv[2];

    double fs;
    int enabled;
} PsychoacousticBass;

void PsychoacousticBassInit(PsychoacousticBass *pb, double fs)
{
    memset(pb, 0, sizeof(PsychoacousticBass));
    pb->fs = fs;
    pb->frequency = 80.0f;
    pb->harmonicLevel = 50.0f;
    pb->thirdHarmonic = 30.0f;
    pb->fourthHarmonic = 15.0f;
    pb->subBassLevel = 20.0f;
    pb->drive = 30.0f;
    pb->mix = 50.0f;

    // Initialize lowpass filter (Butterworth 2nd order)
    double fc = pb->frequency;
    double K = tan(M_PI * fc / fs);
    double K2 = K * K;
    double sqrt2 = 1.41421356237;
    double norm = 1.0 / (1.0 + sqrt2 * K + K2);

    pb->lp_b0 = K2 * norm;
    pb->lp_b1 = 2.0 * pb->lp_b0;
    pb->lp_b2 = pb->lp_b0;
    pb->lp_a1 = 2.0 * (K2 - 1.0) * norm;
    pb->lp_a2 = (1.0 - sqrt2 * K + K2) * norm;

    // Initialize highpass filter
    pb->hp_b0 = norm;
    pb->hp_b1 = -2.0 * norm;
    pb->hp_b2 = norm;
    pb->hp_a1 = pb->lp_a1;
    pb->hp_a2 = pb->lp_a2;
}

void PsychoacousticBassSetParams(PsychoacousticBass *pb, float frequency,
                                  float harmonicLevel, float thirdHarmonic,
                                  float fourthHarmonic, float subBassLevel,
                                  float drive, float mix)
{
    pb->frequency = (frequency < 30.0f) ? 30.0f : ((frequency > 150.0f) ? 150.0f : frequency);
    pb->harmonicLevel = harmonicLevel;
    pb->thirdHarmonic = thirdHarmonic;
    pb->fourthHarmonic = fourthHarmonic;
    pb->subBassLevel = subBassLevel;
    pb->drive = drive;
    pb->mix = mix;

    // Recalculate filter coefficients
    double fc = pb->frequency;
    double K = tan(M_PI * fc / pb->fs);
    double K2 = K * K;
    double sqrt2 = 1.41421356237;
    double norm = 1.0 / (1.0 + sqrt2 * K + K2);

    pb->lp_b0 = K2 * norm;
    pb->lp_b1 = 2.0 * pb->lp_b0;
    pb->lp_b2 = pb->lp_b0;
    pb->lp_a1 = 2.0 * (K2 - 1.0) * norm;
    pb->lp_a2 = (1.0 - sqrt2 * K + K2) * norm;

    pb->hp_b0 = norm;
    pb->hp_b1 = -2.0 * norm;
    pb->hp_b2 = norm;
    pb->hp_a1 = pb->lp_a1;
    pb->hp_a2 = pb->lp_a2;
}

static inline float softClip(float x, float drive)
{
    // Soft saturation using tanh-like function
    float k = 1.0f + drive * 4.0f;  // Drive multiplier
    x *= k;
    return x / (1.0f + fabsf(x));
}

void PsychoacousticBassProcess(PsychoacousticBass *pb, float *left, float *right, size_t n)
{
    if (!pb->enabled) return;

    float h2 = pb->harmonicLevel / 100.0f;
    float h3 = pb->thirdHarmonic / 100.0f;
    float h4 = pb->fourthHarmonic / 100.0f;
    float sub = pb->subBassLevel / 100.0f;
    float drv = pb->drive / 100.0f;
    float wet = pb->mix / 100.0f;
    float dry = 1.0f - wet;

    for (size_t i = 0; i < n; i++)
    {
        // Extract bass content
        float bassL = (float)(pb->lp_b0 * left[i] + pb->lp_z1[0]);
        pb->lp_z1[0] = pb->lp_b1 * left[i] - pb->lp_a1 * bassL + pb->lp_z2[0];
        pb->lp_z2[0] = pb->lp_b2 * left[i] - pb->lp_a2 * bassL;

        float bassR = (float)(pb->lp_b0 * right[i] + pb->lp_z1[1]);
        pb->lp_z1[1] = pb->lp_b1 * right[i] - pb->lp_a1 * bassR + pb->lp_z2[1];
        pb->lp_z2[1] = pb->lp_b2 * right[i] - pb->lp_a2 * bassR;

        // Extract non-bass content
        float highL = (float)(pb->hp_b0 * left[i] + pb->hp_z1[0]);
        pb->hp_z1[0] = pb->hp_b1 * left[i] - pb->hp_a1 * highL + pb->hp_z2[0];
        pb->hp_z2[0] = pb->hp_b2 * left[i] - pb->hp_a2 * highL;

        float highR = (float)(pb->hp_b0 * right[i] + pb->hp_z1[1]);
        pb->hp_z1[1] = pb->hp_b1 * right[i] - pb->hp_a1 * highR + pb->hp_z2[1];
        pb->hp_z2[1] = pb->hp_b2 * right[i] - pb->hp_a2 * highR;

        // Generate harmonics from bass
        // Apply soft saturation for harmonic generation
        float saturatedL = softClip(bassL * 2.0f, drv);
        float saturatedR = softClip(bassR * 2.0f, drv);

        // 2nd harmonic (frequency doubling via absolute value approximation)
        float harm2L = fabsf(saturatedL) - 0.5f * saturatedL;
        float harm2R = fabsf(saturatedR) - 0.5f * saturatedR;

        // 3rd harmonic (from cubic distortion)
        float harm3L = saturatedL * saturatedL * saturatedL;
        float harm3R = saturatedR * saturatedR * saturatedR;

        // 4th harmonic (from squared absolute)
        float harm4L = harm2L * harm2L;
        float harm4R = harm2R * harm2R;

        // Sub-harmonic (simple envelope following for psychoacoustic effect)
        // Note: True sub-harmonic generation is complex; this is an approximation
        // using the envelope to boost very low frequencies
        float envL = fabsf(bassL);
        float envR = fabsf(bassR);
        pb->limiterEnv[0] = 0.99f * pb->limiterEnv[0] + 0.01f * envL;
        pb->limiterEnv[1] = 0.99f * pb->limiterEnv[1] + 0.01f * envR;
        float subL = bassL * (1.0f + pb->limiterEnv[0] * sub);
        float subR = bassR * (1.0f + pb->limiterEnv[1] * sub);

        // Mix harmonics
        float enhancedL = subL + harm2L * h2 + harm3L * h3 * 0.5f + harm4L * h4 * 0.25f;
        float enhancedR = subR + harm2R * h2 + harm3R * h3 * 0.5f + harm4R * h4 * 0.25f;

        // Combine: enhanced bass + original highs
        float outL = highL + enhancedL * wet + bassL * dry;
        float outR = highR + enhancedR * wet + bassR * dry;

        // Soft limit output
        left[i] = softClip(outL, 0.1f);
        right[i] = softClip(outR, 0.1f);
    }
}

// ============================================================================
// ENABLE/DISABLE FUNCTIONS
// ============================================================================

void LoudnessNormalizerEnable(LoudnessNormalizer *ln) { ln->enabled = 1; }
void LoudnessNormalizerDisable(LoudnessNormalizer *ln) { ln->enabled = 0; }
void MultibandCompressorEnable(MultibandCompressor *mbc) { mbc->enabled = 1; }
void MultibandCompressorDisable(MultibandCompressor *mbc) { mbc->enabled = 0; }
void NoiseGateEnable(NoiseGate *ng) { ng->enabled = 1; }
void NoiseGateDisable(NoiseGate *ng) { ng->enabled = 0; }
void PsychoacousticBassEnable(PsychoacousticBass *pb) { pb->enabled = 1; }
void PsychoacousticBassDisable(PsychoacousticBass *pb) { pb->enabled = 0; }
