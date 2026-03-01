/**
 * S24 ULTRA - Dynamic Spectrum Configuration
 *
 * ALL parameters are configurable - nothing hardcoded in shaders.
 * Configuration can come from JSON presets or runtime updates.
 */

#ifndef SPECTRUM_CONFIG_H
#define SPECTRUM_CONFIG_H

#include <cstdint>
#include <array>

namespace vulkan_spectrum {

// ═══════════════════════════════════════════════════════════════════════════
// COLOR DEFINITIONS - All colors are dynamic
// ═══════════════════════════════════════════════════════════════════════════

struct Color3 {
    float r, g, b;

    Color3() : r(0), g(0), b(0) {}
    Color3(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
};

struct Color4 {
    float r, g, b, a;

    Color4() : r(0), g(0), b(0), a(1) {}
    Color4(float r_, float g_, float b_, float a_ = 1.0f) : r(r_), g(g_), b(b_), a(a_) {}
};

// ═══════════════════════════════════════════════════════════════════════════
// THEME CONFIGURATION - Defines visual appearance
// ═══════════════════════════════════════════════════════════════════════════

struct ThemeColors {
    // Gradient colors (low to high intensity)
    Color3 gradientLow;
    Color3 gradientHigh;

    // Sky colors (horizon to zenith)
    Color3 skyHorizon;
    Color3 skyZenith;

    // Material colors
    Color3 terrainGrass;
    Color3 terrainRock;
    Color3 terrainSnow;

    // Accent colors
    Color3 glowColor;
    Color3 sunColor;
    Color3 gridColor;

    // Bar colors (for spectrum bars)
    float barHueStart;      // Starting hue (0-1)
    float barHueRange;      // Hue range to sweep
    float barSaturation;
    float barBrightness;
};

// ═══════════════════════════════════════════════════════════════════════════
// TERRAIN CONFIGURATION - Procedural terrain generation
// ═══════════════════════════════════════════════════════════════════════════

struct TerrainConfig {
    // Noise parameters
    float noiseScale1;          // Primary noise frequency
    float noiseScale2;          // Secondary noise frequency
    float noiseAmplitude1;      // Primary noise amplitude
    float noiseAmplitude2;      // Secondary noise amplitude
    float baseHeight;           // Minimum terrain height

    // FFT influence
    float fftInfluence;         // How much FFT affects height (0-5)
    float fftBinScale;          // Bin mapping scale
    float bassInfluence;        // Additional bass boost factor

    // Material thresholds
    float snowThreshold;        // Height for snow
    float rockThreshold;        // Height for rock

    // Grid overlay
    float gridThickness;        // Grid line thickness (0-0.5)
    float gridBrightness;       // Grid line brightness boost

    // Intersection
    int coarseSteps;            // Binary search coarse steps
    int fineSteps;              // Binary search fine steps
    float maxDistance;          // Max ray distance
    float normalEpsilon;        // Normal calculation epsilon
};

// ═══════════════════════════════════════════════════════════════════════════
// CAMERA CONFIGURATION - View and movement
// ═══════════════════════════════════════════════════════════════════════════

struct CameraConfig {
    // Position
    float height;               // Base camera height
    float distance;             // Distance behind target
    float targetHeight;         // Look-at target height
    float targetDistance;       // Look-at distance ahead

    // Movement
    float moveSpeed;            // Forward movement speed
    float tiltSensitivity;      // Accelerometer sensitivity
    float maxTiltX;             // Max X tilt angle
    float maxTiltY;             // Max Y tilt angle

    // Field of view (for future perspective projection)
    float fov;                  // Field of view in degrees
};

// ═══════════════════════════════════════════════════════════════════════════
// LIGHTING CONFIGURATION - Scene illumination
// ═══════════════════════════════════════════════════════════════════════════

struct LightingConfig {
    // Sun/main light
    float sunDirX, sunDirY, sunDirZ;
    Color3 sunColor;
    float sunIntensity;

    // Ambient
    Color3 ambientColor;
    float ambientIntensity;

    // Sun glow in sky
    float sunGlowPower;         // Falloff exponent
    float sunGlowIntensity;     // Glow brightness

    // FFT glow
    float fftGlowIntensity;     // FFT-reactive glow

    // Fog
    float fogDensity;           // Exponential fog density
    float fogStart;             // Where fog begins
};

// ═══════════════════════════════════════════════════════════════════════════
// BAR CONFIGURATION - Spectrum bar rendering
// ═══════════════════════════════════════════════════════════════════════════

struct BarConfig {
    // Dimensions
    float widthRatio;           // Bar width as ratio of slot (0-1)
    float maxHeight;            // Maximum bar height (0-1)
    float marginLeft;           // Left margin within slot
    float marginRight;          // Right margin within slot

    // Appearance
    float gradientPower;        // Height-based brightness gradient
    float glowRadius;           // Glow around bars
    float glowIntensity;        // Glow brightness

    // Animation
    float attackSpeed;          // How fast bars rise
    float decaySpeed;           // How fast bars fall
    float peakHoldTime;         // Peak indicator hold (seconds)
    float peakDecaySpeed;       // Peak indicator fall speed
};

// ═══════════════════════════════════════════════════════════════════════════
// POST-PROCESSING CONFIGURATION - Visual effects
// ═══════════════════════════════════════════════════════════════════════════

struct PostProcessConfig {
    // Vignette
    float vignetteIntensity;
    float vignetteRadius;
    float vignetteSoftness;

    // Scanlines
    float scanlineIntensity;
    float scanlineCount;        // Lines per screen height
    float scanlineSpeed;        // Scroll speed

    // Chromatic aberration
    float chromaticAmount;
    float chromaticFalloff;     // Edge-based falloff

    // Bloom
    float bloomIntensity;
    float bloomThreshold;
    float bloomRadius;

    // Color grading
    float contrast;
    float saturation;
    float brightness;
    float gamma;

    // Bass pulse
    float bassPulseIntensity;   // Screen pulse on bass
    Color3 bassPulseColor;
};

// ═══════════════════════════════════════════════════════════════════════════
// AUDIO CONFIGURATION - FFT and analysis
// ═══════════════════════════════════════════════════════════════════════════

struct AudioConfig {
    // FFT
    int barCount;               // Number of spectrum bars
    float minDb;                // Minimum dB level
    float maxDb;                // Maximum dB level

    // Smoothing
    float attackSmooth;         // Rising smoothing factor
    float decaySmooth;          // Falling smoothing factor

    // Bass detection
    int bassEndBin;             // Last bin considered "bass"
    float bassBoost;            // Bass level multiplier
};

// ═══════════════════════════════════════════════════════════════════════════
// COMPLETE SPECTRUM CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

struct SpectrumConfig {
    // Metadata
    char name[64];
    char author[64];
    int version;

    // Sub-configurations
    ThemeColors theme;
    TerrainConfig terrain;
    CameraConfig camera;
    LightingConfig lighting;
    BarConfig bars;
    PostProcessConfig postProcess;
    AudioConfig audio;

    // Runtime state (not saved)
    int stereoMode;             // 0=mono, 1=mirror, 2=split
    bool enableTerrain;
    bool enableBars;
    bool enableGrid;
};

// ═══════════════════════════════════════════════════════════════════════════
// GPU UNIFORM BUFFER - Packed for shader access (std140 layout)
// Must be 16-byte aligned for Vulkan
// ═══════════════════════════════════════════════════════════════════════════

struct alignas(16) ShaderUniforms {
    // Resolution & time (16 bytes)
    float resolution[2];
    float time;
    float deltaTime;

    // Camera (16 bytes)
    float tiltX;
    float tiltY;
    float cameraHeight;
    float cameraSpeed;

    // Audio levels (16 bytes)
    float bassLevel;
    float midLevel;
    float highLevel;
    float overallLevel;

    // Audio config (16 bytes)
    int barCount;
    float minDb;
    float maxDb;
    float fftInfluence;

    // Theme gradient (32 bytes)
    float gradientLow[4];       // rgb + padding
    float gradientHigh[4];      // rgb + padding

    // Sky colors (32 bytes)
    float skyHorizon[4];
    float skyZenith[4];

    // Terrain materials (48 bytes)
    float terrainGrass[4];
    float terrainRock[4];
    float terrainSnow[4];

    // Terrain params (16 bytes)
    float noiseScale1;
    float noiseScale2;
    float noiseAmp1;
    float noiseAmp2;

    // Terrain thresholds (16 bytes)
    float baseHeight;
    float snowThreshold;
    float rockThreshold;
    float gridThickness;

    // Lighting sun (16 bytes)
    float sunDir[3];
    float sunIntensity;

    // Lighting colors (32 bytes)
    float sunColor[4];
    float ambientColor[4];

    // Lighting params (16 bytes)
    float ambientIntensity;
    float fogDensity;
    float sunGlowPower;
    float sunGlowIntensity;

    // Bar config (16 bytes)
    float barWidthRatio;
    float barMaxHeight;
    float barMarginLeft;
    float barMarginRight;

    // Bar appearance (16 bytes)
    float barHueStart;
    float barHueRange;
    float barSaturation;
    float barBrightness;

    // Post-process vignette (16 bytes)
    float vignetteIntensity;
    float vignetteRadius;
    float vignetteSoftness;
    float scanlineIntensity;

    // Post-process effects (16 bytes)
    float scanlineCount;
    float chromaticAmount;
    float bloomIntensity;
    float bloomThreshold;

    // Post-process color (16 bytes)
    float contrast;
    float saturation;
    float brightness;
    float gamma;

    // Bass pulse (16 bytes)
    float bassPulseIntensity;
    float bassPulseR;
    float bassPulseG;
    float bassPulseB;

    // Flags (16 bytes)
    int stereoMode;
    int enableTerrain;
    int enableBars;
    int enableGrid;

    // Intersection config (16 bytes)
    int coarseSteps;
    int fineSteps;
    float maxDistance;
    float normalEpsilon;

    // Glow config (16 bytes)
    float glowIntensity;
    float fftGlowIntensity;
    float glowR;
    float glowG;

    // Padding to 512 bytes for alignment
    // Base struct: 26 * 16 = 416 bytes, need 512 - 416 = 96 bytes padding
    float glowB;
    float _padding[23];  // glowB (4) + _padding[23] (92) = 96 bytes
};

static_assert(sizeof(ShaderUniforms) == 512, "ShaderUniforms must be 512 bytes");

// ═══════════════════════════════════════════════════════════════════════════
// DEFAULT CONFIGURATIONS
// ═══════════════════════════════════════════════════════════════════════════

SpectrumConfig getDefaultConfig();
SpectrumConfig getNeonConfig();
SpectrumConfig getFireConfig();
SpectrumConfig getMatrixConfig();
SpectrumConfig getOceanConfig();

// Convert config to shader uniforms
ShaderUniforms configToUniforms(const SpectrumConfig& config, float time, float deltaTime);

} // namespace vulkan_spectrum

#endif // SPECTRUM_CONFIG_H
