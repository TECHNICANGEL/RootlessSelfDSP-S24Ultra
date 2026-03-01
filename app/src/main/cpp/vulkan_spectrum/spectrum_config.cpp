/**
 * S24 ULTRA - Spectrum Configuration Implementation
 *
 * Default presets and conversion utilities
 */

#include "spectrum_config.h"
#include <cstring>
#include <cmath>

namespace vulkan_spectrum {

// ═══════════════════════════════════════════════════════════════════════════
// BASE CONFIGURATION - Shared defaults
// ═══════════════════════════════════════════════════════════════════════════

static SpectrumConfig createBaseConfig() {
    SpectrumConfig config = {};

    strncpy(config.name, "Default", sizeof(config.name));
    strncpy(config.author, "S24 Ultra", sizeof(config.author));
    config.version = 1;

    // Terrain - Lower peaks, better visibility
    config.terrain.noiseScale1 = 0.15f;
    config.terrain.noiseScale2 = 0.3f;
    config.terrain.noiseAmplitude1 = 2.5f;    // Lower base terrain
    config.terrain.noiseAmplitude2 = 1.0f;
    config.terrain.baseHeight = 0.2f;         // Low floor
    config.terrain.fftInfluence = 4.0f;       // Reduced FFT peaks (was 8.0)
    config.terrain.fftBinScale = 2.0f;
    config.terrain.bassInfluence = 1.5f;      // Less bass pump
    config.terrain.snowThreshold = 4.5f;
    config.terrain.rockThreshold = 2.5f;
    config.terrain.gridThickness = 0.44f;
    config.terrain.gridBrightness = 0.2f;
    config.terrain.coarseSteps = 8;
    config.terrain.fineSteps = 4;
    config.terrain.maxDistance = 50.0f;
    config.terrain.normalEpsilon = 0.1f;

    // Camera - Higher view, looking down at terrain
    config.camera.height = 5.0f;              // Higher camera
    config.camera.distance = 4.0f;
    config.camera.targetHeight = 0.5f;        // Look down at terrain
    config.camera.targetDistance = 8.0f;      // Look further ahead
    config.camera.moveSpeed = 0.5f;
    config.camera.tiltSensitivity = 1.5f;
    config.camera.maxTiltX = 1.0f;
    config.camera.maxTiltY = 0.3f;
    config.camera.fov = 60.0f;

    // Lighting
    config.lighting.sunDirX = 0.5f;
    config.lighting.sunDirY = 0.6f;
    config.lighting.sunDirZ = 0.3f;
    config.lighting.sunColor = {1.2f, 1.1f, 1.0f};
    config.lighting.sunIntensity = 1.0f;
    config.lighting.ambientColor = {0.3f, 0.35f, 0.4f};
    config.lighting.ambientIntensity = 1.0f;
    config.lighting.sunGlowPower = 4.0f;
    config.lighting.sunGlowIntensity = 0.5f;
    config.lighting.fftGlowIntensity = 1.0f;   // REACTIVE: Was 0.4, now 2.5x stronger FFT glow
    config.lighting.fogDensity = 0.03f;        // Slightly less fog to see more detail
    config.lighting.fogStart = 0.0f;

    // Bars
    config.bars.widthRatio = 0.7f;
    config.bars.maxHeight = 0.7f;
    config.bars.marginLeft = 0.15f;
    config.bars.marginRight = 0.15f;
    config.bars.gradientPower = 0.6f;
    config.bars.glowRadius = 0.15f;            // REACTIVE: Larger glow radius
    config.bars.glowIntensity = 0.7f;          // REACTIVE: Was 0.3, now stronger glow
    config.bars.attackSpeed = 0.85f;           // REACTIVE: Was 0.4, fast attack
    config.bars.decaySpeed = 0.4f;             // REACTIVE: Was 0.2, medium decay
    config.bars.peakHoldTime = 0.5f;
    config.bars.peakDecaySpeed = 0.5f;

    // Post-processing
    config.postProcess.vignetteIntensity = 0.4f;
    config.postProcess.vignetteRadius = 0.5f;
    config.postProcess.vignetteSoftness = 0.5f;
    config.postProcess.scanlineIntensity = 0.0f;
    config.postProcess.scanlineCount = 240.0f;
    config.postProcess.scanlineSpeed = 0.0f;
    config.postProcess.chromaticAmount = 0.0f;
    config.postProcess.chromaticFalloff = 1.0f;
    config.postProcess.bloomIntensity = 0.0f;
    config.postProcess.bloomThreshold = 0.8f;
    config.postProcess.bloomRadius = 4.0f;
    config.postProcess.contrast = 1.0f;
    config.postProcess.saturation = 1.0f;
    config.postProcess.brightness = 1.0f;
    config.postProcess.gamma = 1.0f;
    config.postProcess.bassPulseIntensity = 0.5f;   // REACTIVE: Was 0.08, now 6x stronger screen pulse!
    config.postProcess.bassPulseColor = {0.6f, 0.4f, 0.8f};  // More saturated pulse color

    // Audio
    config.audio.barCount = 256;
    config.audio.minDb = -70.0f;               // REACTIVE: Was -80, narrower range = more dynamic
    config.audio.maxDb = 0.0f;
    config.audio.attackSmooth = 0.85f;         // REACTIVE: Was 0.4, fast attack
    config.audio.decaySmooth = 0.4f;           // REACTIVE: Was 0.2, medium decay
    config.audio.bassEndBin = 48;              // REACTIVE: Was 32, wider bass range
    config.audio.bassBoost = 1.8f;             // REACTIVE: Was 1.0, 80% bass boost

    // Runtime
    config.stereoMode = 0;
    config.enableTerrain = true;
    config.enableBars = false;
    config.enableGrid = true;

    return config;
}

// ═══════════════════════════════════════════════════════════════════════════
// NEON THEME - Cyberpunk purple/cyan
// ═══════════════════════════════════════════════════════════════════════════

SpectrumConfig getNeonConfig() {
    SpectrumConfig config = createBaseConfig();

    strncpy(config.name, "Neon", sizeof(config.name));

    // Theme colors
    config.theme.gradientLow = {0.1f, 0.0f, 0.3f};
    config.theme.gradientHigh = {0.0f, 1.0f, 1.0f};
    config.theme.skyHorizon = {0.02f, 0.0f, 0.05f};
    config.theme.skyZenith = {0.12f, 0.0f, 0.25f};
    config.theme.terrainGrass = {0.3f, 0.5f, 0.2f};
    config.theme.terrainRock = {0.45f, 0.42f, 0.38f};
    config.theme.terrainSnow = {0.9f, 0.92f, 0.95f};
    config.theme.glowColor = {0.5f, 0.0f, 1.0f};
    config.theme.sunColor = {0.8f, 0.4f, 1.0f};
    config.theme.gridColor = {0.3f, 0.0f, 0.5f};
    config.theme.barHueStart = 0.75f;
    config.theme.barHueRange = -0.5f;
    config.theme.barSaturation = 0.8f;
    config.theme.barBrightness = 0.9f;

    // Neon-specific post-processing
    config.postProcess.bassPulseColor = {0.3f, 0.0f, 0.5f};

    return config;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIRE THEME - Red/orange flames
// ═══════════════════════════════════════════════════════════════════════════

SpectrumConfig getFireConfig() {
    SpectrumConfig config = createBaseConfig();

    strncpy(config.name, "Fire", sizeof(config.name));

    // Theme colors
    config.theme.gradientLow = {0.3f, 0.0f, 0.0f};
    config.theme.gradientHigh = {1.0f, 0.5f, 0.0f};
    config.theme.skyHorizon = {0.1f, 0.02f, 0.0f};
    config.theme.skyZenith = {0.4f, 0.15f, 0.0f};
    config.theme.terrainGrass = {0.4f, 0.3f, 0.1f};
    config.theme.terrainRock = {0.3f, 0.2f, 0.15f};
    config.theme.terrainSnow = {0.95f, 0.9f, 0.85f};
    config.theme.glowColor = {1.0f, 0.3f, 0.0f};
    config.theme.sunColor = {1.0f, 0.6f, 0.2f};
    config.theme.gridColor = {0.5f, 0.2f, 0.0f};
    config.theme.barHueStart = 0.0f;
    config.theme.barHueRange = 0.12f;
    config.theme.barSaturation = 0.9f;
    config.theme.barBrightness = 0.95f;

    // Fire-specific
    config.postProcess.bassPulseColor = {0.6f, 0.2f, 0.0f};
    config.lighting.sunColor = {1.4f, 1.0f, 0.6f};

    return config;
}

// ═══════════════════════════════════════════════════════════════════════════
// MATRIX THEME - Green digital rain
// ═══════════════════════════════════════════════════════════════════════════

SpectrumConfig getMatrixConfig() {
    SpectrumConfig config = createBaseConfig();

    strncpy(config.name, "Matrix", sizeof(config.name));

    // Theme colors
    config.theme.gradientLow = {0.0f, 0.1f, 0.0f};
    config.theme.gradientHigh = {0.0f, 1.0f, 0.3f};
    config.theme.skyHorizon = {0.0f, 0.02f, 0.01f};
    config.theme.skyZenith = {0.0f, 0.1f, 0.03f};
    config.theme.terrainGrass = {0.1f, 0.4f, 0.15f};
    config.theme.terrainRock = {0.2f, 0.35f, 0.2f};
    config.theme.terrainSnow = {0.7f, 0.95f, 0.75f};
    config.theme.glowColor = {0.0f, 1.0f, 0.2f};
    config.theme.sunColor = {0.4f, 1.0f, 0.5f};
    config.theme.gridColor = {0.0f, 0.4f, 0.1f};
    config.theme.barHueStart = 0.28f;
    config.theme.barHueRange = 0.08f;
    config.theme.barSaturation = 0.85f;
    config.theme.barBrightness = 0.9f;

    // Matrix-specific - digital scanlines
    config.postProcess.scanlineIntensity = 0.15f;
    config.postProcess.scanlineCount = 320.0f;
    config.postProcess.bassPulseColor = {0.0f, 0.5f, 0.1f};

    return config;
}

// ═══════════════════════════════════════════════════════════════════════════
// OCEAN THEME - Deep blue waves
// ═══════════════════════════════════════════════════════════════════════════

SpectrumConfig getOceanConfig() {
    SpectrumConfig config = createBaseConfig();

    strncpy(config.name, "Ocean", sizeof(config.name));

    // Theme colors
    config.theme.gradientLow = {0.0f, 0.1f, 0.2f};
    config.theme.gradientHigh = {0.0f, 0.7f, 1.0f};
    config.theme.skyHorizon = {0.0f, 0.03f, 0.08f};
    config.theme.skyZenith = {0.0f, 0.12f, 0.25f};
    config.theme.terrainGrass = {0.2f, 0.4f, 0.35f};
    config.theme.terrainRock = {0.3f, 0.4f, 0.45f};
    config.theme.terrainSnow = {0.85f, 0.95f, 1.0f};
    config.theme.glowColor = {0.0f, 0.5f, 1.0f};
    config.theme.sunColor = {0.6f, 0.9f, 1.0f};
    config.theme.gridColor = {0.0f, 0.3f, 0.5f};
    config.theme.barHueStart = 0.5f;
    config.theme.barHueRange = 0.15f;
    config.theme.barSaturation = 0.75f;
    config.theme.barBrightness = 0.85f;

    // Ocean-specific - wavy, calmer
    config.postProcess.bassPulseColor = {0.0f, 0.3f, 0.5f};
    config.lighting.ambientColor = {0.25f, 0.35f, 0.45f};

    return config;
}

// ═══════════════════════════════════════════════════════════════════════════
// DEFAULT - Returns Neon as default
// ═══════════════════════════════════════════════════════════════════════════

SpectrumConfig getDefaultConfig() {
    return getNeonConfig();
}

// ═══════════════════════════════════════════════════════════════════════════
// CONVERT CONFIG TO SHADER UNIFORMS
// ═══════════════════════════════════════════════════════════════════════════

ShaderUniforms configToUniforms(const SpectrumConfig& config, float time, float deltaTime) {
    ShaderUniforms u = {};

    // Time
    u.time = time;
    u.deltaTime = deltaTime;

    // Camera
    u.cameraHeight = config.camera.height;
    u.cameraSpeed = config.camera.moveSpeed;

    // Audio
    u.barCount = config.audio.barCount;
    u.minDb = config.audio.minDb;
    u.maxDb = config.audio.maxDb;
    u.fftInfluence = config.terrain.fftInfluence;

    // Theme gradient
    u.gradientLow[0] = config.theme.gradientLow.r;
    u.gradientLow[1] = config.theme.gradientLow.g;
    u.gradientLow[2] = config.theme.gradientLow.b;
    u.gradientLow[3] = 1.0f;

    u.gradientHigh[0] = config.theme.gradientHigh.r;
    u.gradientHigh[1] = config.theme.gradientHigh.g;
    u.gradientHigh[2] = config.theme.gradientHigh.b;
    u.gradientHigh[3] = 1.0f;

    // Sky
    u.skyHorizon[0] = config.theme.skyHorizon.r;
    u.skyHorizon[1] = config.theme.skyHorizon.g;
    u.skyHorizon[2] = config.theme.skyHorizon.b;
    u.skyHorizon[3] = 1.0f;

    u.skyZenith[0] = config.theme.skyZenith.r;
    u.skyZenith[1] = config.theme.skyZenith.g;
    u.skyZenith[2] = config.theme.skyZenith.b;
    u.skyZenith[3] = 1.0f;

    // Terrain materials
    u.terrainGrass[0] = config.theme.terrainGrass.r;
    u.terrainGrass[1] = config.theme.terrainGrass.g;
    u.terrainGrass[2] = config.theme.terrainGrass.b;
    u.terrainGrass[3] = 1.0f;

    u.terrainRock[0] = config.theme.terrainRock.r;
    u.terrainRock[1] = config.theme.terrainRock.g;
    u.terrainRock[2] = config.theme.terrainRock.b;
    u.terrainRock[3] = 1.0f;

    u.terrainSnow[0] = config.theme.terrainSnow.r;
    u.terrainSnow[1] = config.theme.terrainSnow.g;
    u.terrainSnow[2] = config.theme.terrainSnow.b;
    u.terrainSnow[3] = 1.0f;

    // Terrain params
    u.noiseScale1 = config.terrain.noiseScale1;
    u.noiseScale2 = config.terrain.noiseScale2;
    u.noiseAmp1 = config.terrain.noiseAmplitude1;
    u.noiseAmp2 = config.terrain.noiseAmplitude2;
    u.baseHeight = config.terrain.baseHeight;
    u.snowThreshold = config.terrain.snowThreshold;
    u.rockThreshold = config.terrain.rockThreshold;
    u.gridThickness = config.terrain.gridThickness;

    // Lighting
    float sunLen = sqrtf(config.lighting.sunDirX * config.lighting.sunDirX +
                         config.lighting.sunDirY * config.lighting.sunDirY +
                         config.lighting.sunDirZ * config.lighting.sunDirZ);
    if (sunLen > 0.001f) {
        u.sunDir[0] = config.lighting.sunDirX / sunLen;
        u.sunDir[1] = config.lighting.sunDirY / sunLen;
        u.sunDir[2] = config.lighting.sunDirZ / sunLen;
    }
    u.sunIntensity = config.lighting.sunIntensity;

    u.sunColor[0] = config.lighting.sunColor.r;
    u.sunColor[1] = config.lighting.sunColor.g;
    u.sunColor[2] = config.lighting.sunColor.b;
    u.sunColor[3] = 1.0f;

    u.ambientColor[0] = config.lighting.ambientColor.r;
    u.ambientColor[1] = config.lighting.ambientColor.g;
    u.ambientColor[2] = config.lighting.ambientColor.b;
    u.ambientColor[3] = 1.0f;

    u.ambientIntensity = config.lighting.ambientIntensity;
    u.fogDensity = config.lighting.fogDensity;
    u.sunGlowPower = config.lighting.sunGlowPower;
    u.sunGlowIntensity = config.lighting.sunGlowIntensity;

    // Bars
    u.barWidthRatio = config.bars.widthRatio;
    u.barMaxHeight = config.bars.maxHeight;
    u.barMarginLeft = config.bars.marginLeft;
    u.barMarginRight = config.bars.marginRight;
    u.barHueStart = config.theme.barHueStart;
    u.barHueRange = config.theme.barHueRange;
    u.barSaturation = config.theme.barSaturation;
    u.barBrightness = config.theme.barBrightness;

    // Post-process
    u.vignetteIntensity = config.postProcess.vignetteIntensity;
    u.vignetteRadius = config.postProcess.vignetteRadius;
    u.vignetteSoftness = config.postProcess.vignetteSoftness;
    u.scanlineIntensity = config.postProcess.scanlineIntensity;
    u.scanlineCount = config.postProcess.scanlineCount;
    u.chromaticAmount = config.postProcess.chromaticAmount;
    u.bloomIntensity = config.postProcess.bloomIntensity;
    u.bloomThreshold = config.postProcess.bloomThreshold;
    u.contrast = config.postProcess.contrast;
    u.saturation = config.postProcess.saturation;
    u.brightness = config.postProcess.brightness;
    u.gamma = config.postProcess.gamma;
    u.bassPulseIntensity = config.postProcess.bassPulseIntensity;
    u.bassPulseR = config.postProcess.bassPulseColor.r;
    u.bassPulseG = config.postProcess.bassPulseColor.g;
    u.bassPulseB = config.postProcess.bassPulseColor.b;

    // Flags
    u.stereoMode = config.stereoMode;
    u.enableTerrain = config.enableTerrain ? 1 : 0;
    u.enableBars = config.enableBars ? 1 : 0;
    u.enableGrid = config.enableGrid ? 1 : 0;

    // Intersection
    u.coarseSteps = config.terrain.coarseSteps;
    u.fineSteps = config.terrain.fineSteps;
    u.maxDistance = config.terrain.maxDistance;
    u.normalEpsilon = config.terrain.normalEpsilon;

    // Glow
    u.glowIntensity = config.bars.glowIntensity;
    u.fftGlowIntensity = config.lighting.fftGlowIntensity;
    u.glowR = config.theme.glowColor.r;
    u.glowG = config.theme.glowColor.g;
    u.glowB = config.theme.glowColor.b;

    return u;
}

} // namespace vulkan_spectrum
