#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../jdsp_header.h"
void StereoEnhancementRefresh(JamesDSPLib *jdsp)
{
  jdsp_lock(jdsp);
	WarpedPFB *subband0 = (WarpedPFB *)jdsp->sterEnh.subband[0];
	WarpedPFB *subband1 = (WarpedPFB *)jdsp->sterEnh.subband[1];
	initWarpedPFB(subband0, jdsp->fs, 5, 2);
	assignPtrWarpedPFB(subband1, 5, 2);
	float ms = 1.2f; // 1.2 ms
	for (unsigned int i = 0; i < 5; i++)
		jdsp->sterEnh.emaAlpha[i] = 1.0f - powf(10.0f, (log10f(0.5f) / (ms / 1000.0f) / (jdsp->fs / (float)subband0->Sk[i])));
  jdsp_unlock(jdsp);
}
void StereoEnhancementSetParam(JamesDSPLib *jdsp, float mix)
{
  jdsp_lock(jdsp);
	jdsp->sterEnh.mix = mix;
	jdsp->sterEnh.minusMix = 1.0f - jdsp->sterEnh.mix;
	if (jdsp->sterEnh.mix > 0.5f)
		jdsp->sterEnh.gain = 3.0f - jdsp->sterEnh.mix * 2.0f;
	else
		jdsp->sterEnh.gain = jdsp->sterEnh.mix * 2.0f + 1.0f;
  jdsp_unlock(jdsp);
}

void StereoEnhancementSetDepth(JamesDSPLib *jdsp, float depth)
{
	jdsp_lock(jdsp);
	// Clamp depth between 0.5 and 3.0
	if (depth < 0.5f) depth = 0.5f;
	if (depth > 3.0f) depth = 3.0f;
	jdsp->sterEnh.depth = depth;
	jdsp_unlock(jdsp);
}

void StereoEnhancementSetCenterLevel(JamesDSPLib *jdsp, float level)
{
	jdsp_lock(jdsp);
	// Clamp level between 0.0 and 1.0
	if (level < 0.0f) level = 0.0f;
	if (level > 1.0f) level = 1.0f;
	jdsp->sterEnh.centerLevel = level;
	jdsp_unlock(jdsp);
}

void StereoEnhancementSetBandMix(JamesDSPLib *jdsp, float lowMix, float midMix, float highMix)
{
	jdsp_lock(jdsp);
	// Clamp all values between 0.0 and 1.0
	if (lowMix < 0.0f) lowMix = 0.0f;
	if (lowMix > 1.0f) lowMix = 1.0f;
	if (midMix < 0.0f) midMix = 0.0f;
	if (midMix > 1.0f) midMix = 1.0f;
	if (highMix < 0.0f) highMix = 0.0f;
	if (highMix > 1.0f) highMix = 1.0f;
	jdsp->sterEnh.lowFreqMix = lowMix;
	jdsp->sterEnh.midFreqMix = midMix;
	jdsp->sterEnh.highFreqMix = highMix;
	jdsp_unlock(jdsp);
}
void StereoEnhancementConstructor(JamesDSPLib *jdsp)
{
	StereoEnhancementRefresh(jdsp);
	// Initialize enhanced parameters with defaults
	jdsp->sterEnh.depth = 1.0f;
	jdsp->sterEnh.centerLevel = 0.5f;
	jdsp->sterEnh.lowFreqMix = 0.3f;   // Keep bass more mono
	jdsp->sterEnh.midFreqMix = 1.0f;   // Full effect on mids
	jdsp->sterEnh.highFreqMix = 1.0f;  // Full effect on highs
}
void StereoEnhancementDestructor(JamesDSPLib *jdsp)
{
}
void StereoEnhancementEnable(JamesDSPLib *jdsp)
{
	jdsp->sterEnhEnabled = 1;
}
void StereoEnhancementDisable(JamesDSPLib *jdsp)
{
	jdsp->sterEnhEnabled = 0;
}
void StereoEnhancementProcess(JamesDSPLib *jdsp, size_t n)
{
	stereoEnhancement *snh = &jdsp->sterEnh;
	WarpedPFB *subband0 = (WarpedPFB *)snh->subband[0];
	WarpedPFB *subband1 = (WarpedPFB *)snh->subband[1];
	unsigned int *samplingPeriod = subband0->decimationCounter;
	unsigned int *Sk = subband0->Sk;
	float *bandLeft = subband0->subbandData;
	float *bandRight = subband1->subbandData;
	float y1, y2;

	// Pre-calculate band mix factors (5 bands: low, low-mid, mid, mid-high, high)
	float bandMix[5];
	bandMix[0] = snh->lowFreqMix;                                    // ~20-200 Hz (bass)
	bandMix[1] = snh->lowFreqMix * 0.5f + snh->midFreqMix * 0.5f;   // ~200-800 Hz
	bandMix[2] = snh->midFreqMix;                                    // ~800-3000 Hz (mids)
	bandMix[3] = snh->midFreqMix * 0.5f + snh->highFreqMix * 0.5f;  // ~3000-8000 Hz
	bandMix[4] = snh->highFreqMix;                                   // ~8000-20000 Hz (highs)

	for (size_t i = 0; i < n; i++)
	{
		analysisWarpedPFBStereo(subband0, subband1, &jdsp->tmpBuffer[0][i], &jdsp->tmpBuffer[1][i]);
		for (int j = 0; j < 5; j++)
		{
			if (samplingPeriod[j] == Sk[j])
			{
				float sum = bandLeft[j] + bandRight[j];
				float diff = bandLeft[j] - bandRight[j];
				float sumSq = sum * sum;
				float diffSq = diff * diff;
				snh->sumStates[j] = snh->sumStates[j] * (1.0f - snh->emaAlpha[j]) + sumSq * snh->emaAlpha[j];
				snh->diffStates[j] = snh->diffStates[j] * (1.0f - snh->emaAlpha[j]) + diffSq * snh->emaAlpha[j];

				// Extract center channel
				float centre = 0.0f;
				if (sumSq > FLT_EPSILON)
					centre = (0.5f - sqrtf(snh->diffStates[j] / snh->sumStates[j]) * 0.5f) * sum;

				// Apply center preservation
				float centerPreserve = centre * snh->centerLevel;

				// Apply per-band stereo width with depth multiplier
				float effectiveMix = snh->mix * bandMix[j] * snh->depth;
				if (effectiveMix > 1.0f) effectiveMix = 1.0f;
				float effectiveMinusMix = 1.0f - effectiveMix;

				// Calculate widened signal
				float widenedLeft = (bandLeft[j] - centre) * effectiveMix + centre * effectiveMinusMix;
				float widenedRight = (bandRight[j] - centre) * effectiveMix + centre * effectiveMinusMix;

				// Blend with center preservation
				bandLeft[j] = widenedLeft + centerPreserve * (1.0f - bandMix[j]);
				bandRight[j] = widenedRight + centerPreserve * (1.0f - bandMix[j]);
			}
		}
		synthesisWarpedPFBStereo(subband0, subband1, &y1, &y2);
		jdsp->tmpBuffer[0][i] = y1 * snh->gain;
		jdsp->tmpBuffer[1][i] = y2 * snh->gain;
	}
}
