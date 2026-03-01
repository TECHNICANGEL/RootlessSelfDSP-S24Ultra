/**
 * Vacuum Tube Analog Modeling - S24 Ultra Enhanced Version
 *
 * Realistic tube saturation with:
 * - Asymmetric soft clipping (even harmonics = warmth)
 * - Symmetric soft clipping (odd harmonics = grit)
 * - DC blocking after asymmetric processing
 * - High frequency softening under drive
 * - Oversampling for anti-aliasing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "../jdsp_header.h"

// Fast tanh approximation (Pade approximant)
static inline double fast_tanh(double x)
{
    if (x < -3.0) return -1.0;
    if (x > 3.0) return 1.0;
    double x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}

// Asymmetric waveshaper - generates even harmonics
static inline double asymmetric_waveshape(double x, double bias)
{
    // Shift the signal to create asymmetry
    double shifted = x + bias * 0.5;
    // Soft clip
    double shaped = fast_tanh(shifted);
    // Remove DC offset introduced by bias
    return shaped - fast_tanh(bias * 0.5);
}

// Symmetric waveshaper - generates odd harmonics
static inline double symmetric_waveshape(double x, double drive)
{
    return fast_tanh(x * (1.0 + drive * 3.0));
}

// Simple 1-pole highpass for DC blocking (~10Hz)
static inline double dc_block(double input, double *state, double coeff)
{
    double output = input - *state;
    *state = input - output * coeff;
    return output;
}

// Simple 1-pole lowpass for HF softening
static inline double lp_filter(double input, double *state, double coeff)
{
    *state += coeff * (input - *state);
    return *state;
}

void VTInit(VacuumTube *tb, double fs)
{
    tb->fs = fs;
    tb->drive = 0.3f;       // 30% default drive
    tb->bias = 0.2f;        // Slight asymmetry
    tb->mix = 0.5f;         // 50% wet
    tb->evenHarm = 0.6f;    // Even harmonics (warmth)
    tb->oddHarm = 0.4f;     // Odd harmonics (grit)
    tb->inputGain = 1.0f;
    tb->outputGain = 1.0f;
    tb->hfSoftFreq = 8000.0; // Start rolling off at 8kHz

    tb->needOversample = 0;
    if (fs >= 30000.0 && fs < 65000.0)
    {
        oversample_makeSmp(&tb->smp[0], 2);
        oversample_makeSmp(&tb->smp[1], 2);
        tb->needOversample = 1;
    }
    else if (fs >= 20000.0 && fs < 30000.0)
    {
        oversample_makeSmp(&tb->smp[0], 3);
        oversample_makeSmp(&tb->smp[1], 3);
        tb->needOversample = 1;
    }
    else if (fs >= 14000.0 && fs < 20000.0)
    {
        oversample_makeSmp(&tb->smp[0], 4);
        oversample_makeSmp(&tb->smp[1], 4);
        tb->needOversample = 1;
    }
    else if (fs < 14000.0)
    {
        oversample_makeSmp(&tb->smp[0], 5);
        oversample_makeSmp(&tb->smp[1], 5);
        tb->needOversample = 1;
    }

    // Clear filter states
    memset(tb->dcBlockState, 0, sizeof(tb->dcBlockState));
    memset(tb->hfSoftState, 0, sizeof(tb->hfSoftState));
}

// Process a single sample through tube saturation
static inline double processTubeSample(double input, double drive, double bias,
                                        double evenHarm, double oddHarm)
{
    // Scale input by drive
    double driven = input * (1.0 + drive * 4.0);

    // Generate even harmonics (asymmetric - warmth)
    double evenOut = asymmetric_waveshape(driven, bias);

    // Generate odd harmonics (symmetric - grit)
    double oddOut = symmetric_waveshape(driven, drive);

    // Mix even and odd harmonics
    double mixed = evenOut * evenHarm + oddOut * oddHarm;

    // Normalize
    double harmSum = evenHarm + oddHarm;
    if (harmSum > 0.0)
        mixed /= harmSum;

    return mixed;
}

void VTProcess(VacuumTube *tb, float *x1, float *x2, float *out1, float *out2, size_t n)
{
    float upsample[2][5];

    // Calculate filter coefficients
    double dcBlockCoeff = 0.995; // ~10Hz highpass
    double effectiveFs = tb->fs;
    if (tb->needOversample)
        effectiveFs *= tb->smp[0].factor;

    // HF softening coefficient - more aggressive with higher drive
    double hfCutoff = tb->hfSoftFreq * (1.0 - tb->drive * 0.5);
    double hfCoeff = 1.0 - exp(-2.0 * M_PI * hfCutoff / effectiveFs);

    if (tb->needOversample)
    {
        for (size_t i = 0; i < n; i++)
        {
            double dry1 = x1[i];
            double dry2 = x2[i];

            // Upsample
            oversample_stepupSmp(&tb->smp[0], (float)(dry1 * tb->inputGain), upsample[0]);
            oversample_stepupSmp(&tb->smp[1], (float)(dry2 * tb->inputGain), upsample[1]);

            // Process each oversampled point
            for (int j = 0; j < tb->smp[0].factor; j++)
            {
                // Tube saturation
                double wet1 = processTubeSample(upsample[0][j], tb->drive, tb->bias,
                                                 tb->evenHarm, tb->oddHarm);
                double wet2 = processTubeSample(upsample[1][j], tb->drive, tb->bias,
                                                 tb->evenHarm, tb->oddHarm);

                // DC blocking
                wet1 = dc_block(wet1, &tb->dcBlockState[0][0], dcBlockCoeff);
                wet2 = dc_block(wet2, &tb->dcBlockState[1][0], dcBlockCoeff);

                // HF softening (more with higher drive)
                if (tb->drive > 0.1)
                {
                    wet1 = lp_filter(wet1, &tb->hfSoftState[0][0], hfCoeff);
                    wet2 = lp_filter(wet2, &tb->hfSoftState[1][0], hfCoeff);
                }

                upsample[0][j] = (float)wet1;
                upsample[1][j] = (float)wet2;
            }

            // Downsample
            double wet1 = oversample_stepdownSmpFloat(&tb->smp[0], upsample[0]);
            double wet2 = oversample_stepdownSmpFloat(&tb->smp[1], upsample[1]);

            // Mix dry/wet and apply output gain
            out1[i] = (float)((dry1 * (1.0 - tb->mix) + wet1 * tb->mix) * tb->outputGain);
            out2[i] = (float)((dry2 * (1.0 - tb->mix) + wet2 * tb->mix) * tb->outputGain);
        }
    }
    else
    {
        // No oversampling path
        for (size_t i = 0; i < n; i++)
        {
            double dry1 = x1[i];
            double dry2 = x2[i];

            // Tube saturation
            double wet1 = processTubeSample(dry1 * tb->inputGain, tb->drive, tb->bias,
                                             tb->evenHarm, tb->oddHarm);
            double wet2 = processTubeSample(dry2 * tb->inputGain, tb->drive, tb->bias,
                                             tb->evenHarm, tb->oddHarm);

            // DC blocking
            wet1 = dc_block(wet1, &tb->dcBlockState[0][0], dcBlockCoeff);
            wet2 = dc_block(wet2, &tb->dcBlockState[1][0], dcBlockCoeff);

            // HF softening
            if (tb->drive > 0.1)
            {
                wet1 = lp_filter(wet1, &tb->hfSoftState[0][0], hfCoeff);
                wet2 = lp_filter(wet2, &tb->hfSoftState[1][0], hfCoeff);
            }

            // Mix dry/wet and apply output gain
            out1[i] = (float)((dry1 * (1.0 - tb->mix) + wet1 * tb->mix) * tb->outputGain);
            out2[i] = (float)((dry2 * (1.0 - tb->mix) + wet2 * tb->mix) * tb->outputGain);
        }
    }
}

void VacuumTubeEnable(JamesDSPLib *jdsp)
{
    VTInit(&jdsp->tube, jdsp->fs);
    jdsp->tubeEnabled = 1;
}

void VacuumTubeDisable(JamesDSPLib *jdsp)
{
    jdsp->tubeEnabled = 0;
}

void VacuumTubeSetDrive(JamesDSPLib *jdsp, float drive)
{
    // Clamp to 0.0 - 1.0
    if (drive < 0.0f) drive = 0.0f;
    if (drive > 1.0f) drive = 1.0f;
    jdsp->tube.drive = drive;

    // Auto-adjust output gain to compensate for drive increase
    // More drive = more level, so reduce output
    jdsp->tube.outputGain = 1.0f / (1.0f + drive * 0.5f);
}

void VacuumTubeSetBias(JamesDSPLib *jdsp, float bias)
{
    // Clamp to -1.0 to 1.0
    if (bias < -1.0f) bias = -1.0f;
    if (bias > 1.0f) bias = 1.0f;
    jdsp->tube.bias = bias;
}

void VacuumTubeSetMix(JamesDSPLib *jdsp, float mix)
{
    // Clamp to 0.0 - 1.0
    if (mix < 0.0f) mix = 0.0f;
    if (mix > 1.0f) mix = 1.0f;
    jdsp->tube.mix = mix;
}

void VacuumTubeSetEvenHarm(JamesDSPLib *jdsp, float level)
{
    // Clamp to 0.0 - 1.0
    if (level < 0.0f) level = 0.0f;
    if (level > 1.0f) level = 1.0f;
    jdsp->tube.evenHarm = level;
}

void VacuumTubeSetOddHarm(JamesDSPLib *jdsp, float level)
{
    // Clamp to 0.0 - 1.0
    if (level < 0.0f) level = 0.0f;
    if (level > 1.0f) level = 1.0f;
    jdsp->tube.oddHarm = level;
}

void VacuumTubeProcess(JamesDSPLib *jdsp, size_t n)
{
    VTProcess(&jdsp->tube, jdsp->tmpBuffer[0], jdsp->tmpBuffer[1],
              jdsp->tmpBuffer[0], jdsp->tmpBuffer[1], n);
}
