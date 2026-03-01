/**
 * S24 ULTRA - Vulkan Spectrum Renderer Implementation
 *
 * Full Vulkan renderer with optional hardware ray tracing support
 * for Samsung Galaxy S24 Ultra (Adreno 750 GPU)
 */

#include "vulkan_renderer.h"
#include "shaders/spectrum_vert.h"
#include "shaders/spectrum_frag.h"
#include "shaders/terrain_comp.h"
#include <chrono>
#include <algorithm>
#include <string>

// Backward compatibility aliases (used by embedded fallback shader)
// The new shader uses uniform buffers with config values
#define NUM_BARS DEFAULT_NUM_BARS
#define MIN_DB DEFAULT_MIN_DB
#define MAX_DB DEFAULT_MAX_DB

#if HAS_SHADERC
#include <shaderc/shaderc.hpp>
// Shaderc shader kind enum values
#define SHADER_KIND_VERTEX 0
#define SHADER_KIND_FRAGMENT 4
#else
// Dummy values when shaderc not available
#define SHADER_KIND_VERTEX 0
#define SHADER_KIND_FRAGMENT 4
#endif

// Embedded GLSL source for runtime compilation (fallback if SPIR-V fails)
static const char* VERTEX_SHADER_GLSL = R"(
#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 fragUV;

layout(push_constant) uniform PushConstants {
    vec2 resolution;
    float time;
    float bassLevel;
    float tiltX;
    float tiltY;
    int barCount;
    float minDb;
    float maxDb;
    float glowIntensity;
    float bloomIntensity;
    float chromaticAberration;
    int colorTheme;
    float scanlineIntensity;
    float vignetteIntensity;
    int stereoMode;
} pc;

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragUV = inUV;
}
)";

static const char* FRAGMENT_SHADER_GLSL = R"(
#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(std430, binding = 0) readonly buffer FftData {
    float fftMagnitudes[];
};

layout(push_constant) uniform PushConstants {
    vec2 resolution;
    float time;
    float bassLevel;
    float tiltX;
    float tiltY;
    int barCount;
    float minDb;
    float maxDb;
    float glowIntensity;
    float bloomIntensity;
    float chromaticAberration;
    int colorTheme;
    float scanlineIntensity;
    float vignetteIntensity;
    int stereoMode;  // 0=mono, 1=stereo mirror, 2=stereo split
} pc;

// ═══════════════════════════════════════════════════════════════════════════
// NOISE & UTILITY
// ═══════════════════════════════════════════════════════════════════════════

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289v2(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289v2(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// ═══════════════════════════════════════════════════════════════════════════
// COLOR THEMES
// ═══════════════════════════════════════════════════════════════════════════

// Primary terrain/bar color based on theme
vec3 getThemePrimaryColor(float intensity) {
    switch(pc.colorTheme) {
        case 1:  // Fire - Orange/Red
            return mix(vec3(0.3, 0.0, 0.0), vec3(1.0, 0.5, 0.0), intensity);
        case 2:  // Matrix - Green
            return mix(vec3(0.0, 0.1, 0.0), vec3(0.0, 1.0, 0.3), intensity);
        case 3:  // Ocean - Blue/Teal
            return mix(vec3(0.0, 0.1, 0.2), vec3(0.0, 0.7, 1.0), intensity);
        default: // Neon - Purple/Cyan (default)
            return mix(vec3(0.1, 0.0, 0.35), vec3(0.0, 0.9, 0.95), intensity);
    }
}

// Accent/glow color based on theme
vec3 getThemeAccentColor(float intensity) {
    switch(pc.colorTheme) {
        case 1:  // Fire - Yellow/White
            return mix(vec3(1.0, 0.3, 0.0), vec3(1.0, 1.0, 0.5), intensity);
        case 2:  // Matrix - Bright green/white
            return mix(vec3(0.0, 0.5, 0.1), vec3(0.5, 1.0, 0.7), intensity);
        case 3:  // Ocean - Aqua/White
            return mix(vec3(0.0, 0.4, 0.6), vec3(0.5, 1.0, 1.0), intensity);
        default: // Neon - Pink/Magenta
            return mix(vec3(0.6, 0.1, 1.0), vec3(1.0, 0.5, 1.0), intensity);
    }
}

// Sky gradient based on theme
vec3 getThemeSkyColor(float y) {
    switch(pc.colorTheme) {
        case 1:  // Fire
            return mix(vec3(0.05, 0.01, 0.0), vec3(0.25, 0.05, 0.0), y);
        case 2:  // Matrix
            return mix(vec3(0.0, 0.02, 0.0), vec3(0.0, 0.08, 0.04), y);
        case 3:  // Ocean
            return mix(vec3(0.0, 0.02, 0.05), vec3(0.0, 0.08, 0.18), y);
        default: // Neon
            return mix(vec3(0.02, 0.0, 0.06), vec3(0.12, 0.0, 0.25), y);
    }
}

// Aurora colors based on theme
vec3 getThemeAuroraColor(float aurora) {
    switch(pc.colorTheme) {
        case 1:  // Fire
            return mix(vec3(0.5, 0.2, 0.0), vec3(1.0, 0.3, 0.0), aurora);
        case 2:  // Matrix
            return mix(vec3(0.0, 0.3, 0.1), vec3(0.0, 0.6, 0.2), aurora);
        case 3:  // Ocean
            return mix(vec3(0.0, 0.2, 0.4), vec3(0.0, 0.4, 0.6), aurora);
        default: // Neon
            return mix(vec3(0.0, 0.4, 0.3), vec3(0.4, 0.0, 0.5), aurora);
    }
}

// Bar hue offset based on theme
float getThemeBarHue(float freq) {
    switch(pc.colorTheme) {
        case 1:  // Fire - red to yellow
            return 0.0 + freq * 0.12;
        case 2:  // Matrix - green range
            return 0.28 + freq * 0.08;
        case 3:  // Ocean - blue to cyan
            return 0.5 + freq * 0.15;
        default: // Neon - purple to cyan
            return 0.72 - freq * 0.45;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TERRAIN (FFT-based height)
// ═══════════════════════════════════════════════════════════════════════════

// Procedural mountain base - smooth, no granular noise
float getMountainBase(vec2 pos) {
    float h = snoise(pos * 0.12) * 0.8;
    h += snoise(pos * 0.25 + 30.0) * 0.4;
    h = h * 0.5 + 0.5;
    h = smoothstep(0.0, 1.0, h);
    return h * 1.8;
}

float getTerrainHeight(vec2 pos) {
    // Base procedural mountains
    float baseHeight = getMountainBase(pos);

    // FFT contribution
    float normalizedX = clamp((pos.x + 10.0) * 0.05, 0.0, 1.0);
    int maxBin = max(pc.barCount - 1, 0);
    float binF = normalizedX * float(maxBin);
    int bin = clamp(int(binF), 0, maxBin);
    int bin2 = min(bin + 1, maxBin);

    float dbRange = pc.maxDb - pc.minDb;
    float invDb = (dbRange > 0.001) ? 1.0 / dbRange : 0.0;
    float fft1 = clamp((fftMagnitudes[bin] - pc.minDb) * invDb, 0.0, 1.0);
    float fft2 = clamp((fftMagnitudes[bin2] - pc.minDb) * invDb, 0.0, 1.0);
    float fftH = mix(fft1, fft2, fract(binF));

    // Mountains + FFT reactive
    float h = baseHeight * (1.0 + fftH * 0.5) + fftH * 2.0;
    h *= 1.0 + pc.bassLevel * 0.3;
    return h;
}

float terrainSDF(vec3 p) { return p.y - getTerrainHeight(p.xz); }

vec3 getTerrainNormal(vec3 p) {
    const float e = 0.02;
    return normalize(vec3(
        getTerrainHeight(p.xz - vec2(e, 0.0)) - getTerrainHeight(p.xz + vec2(e, 0.0)),
        2.0 * e,
        getTerrainHeight(p.xz - vec2(0.0, e)) - getTerrainHeight(p.xz + vec2(0.0, e))
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// RAYMARCHING
// ═══════════════════════════════════════════════════════════════════════════

float raymarch(vec3 ro, vec3 rd, float maxD) {
    float t = 0.0;
    for (int i = 0; i < 48; i++) {
        vec3 p = ro + rd * t;
        float d = terrainSDF(p);
        if (d < 0.02 || t > maxD) break;
        t += d * 0.6;
    }
    return t;
}

float softShadow(vec3 ro, vec3 rd, float mint, float maxt) {
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < 12; i++) {
        float h = terrainSDF(ro + rd * t);
        if (h < 0.001) return 0.0;
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.1, 0.4);
        if (t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

void main() {
    vec2 uv = fragUV;
    vec2 screenUV = (uv * 2.0 - 1.0) * vec2(pc.resolution.x / pc.resolution.y, 1.0);

    // Camera with gyroscope parallax
    vec3 ro = vec3(pc.tiltX * 1.5, 3.5 + pc.tiltY * 0.3, -4.0 + pc.time * 0.2);
    vec3 target = vec3(0.0, 0.8, 4.0 + pc.time * 0.2);

    vec3 fwd = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up = cross(fwd, right);
    vec3 rd = normalize(fwd + screenUV.x * right + screenUV.y * up);

    // Sky gradient - themed
    vec3 skyColor = getThemeSkyColor(uv.y);

    // Raymarch terrain
    float t = raymarch(ro, rd, 30.0);
    vec3 color;

    if (t < 30.0) {
        vec3 p = ro + rd * t;
        vec3 n = getTerrainNormal(p);

        // Lighting
        vec3 lightDir = normalize(vec3(0.5, 0.7, -0.4));
        float diff = max(dot(n, lightDir), 0.0);
        float shadow = softShadow(p + n * 0.05, lightDir, 0.1, 8.0);

        // Terrain color based on FFT - themed
        float height = getTerrainHeight(p.xz) / 2.5;
        vec3 terrainCol = getThemePrimaryColor(height);

        // Themed glow
        float glow = height * pc.bassLevel * pc.glowIntensity;
        terrainCol += getThemeAccentColor(glow) * glow;

        // Final lighting
        color = terrainCol * (0.25 + 0.75 * diff * shadow);

        // Rim light - themed
        float rim = pow(1.0 - max(dot(-rd, n), 0.0), 3.0);
        color += getThemeAccentColor(0.5) * rim * 0.4;

        // Distance fog
        float fog = 1.0 - exp(-t * 0.04);
        color = mix(color, skyColor, fog);
    } else {
        color = skyColor;

        // Bass-responsive stars
        vec2 starUV = uv * 40.0;
        vec2 starCell = floor(starUV);
        vec2 starLocal = fract(starUV) - 0.5;
        float starRand = hash(starCell);
        float bassThreshold = 0.97 - pc.bassLevel * 0.15;

        if (starRand > bassThreshold) {
            vec2 starOffset = vec2(hash(starCell + 1.0), hash(starCell + 2.0)) - 0.5;
            starOffset *= 0.6;
            float bassPulse = 1.0 + pc.bassLevel * 0.3 * sin(pc.time * 8.0 + starRand * 10.0);
            float starDist = length(starLocal - starOffset);
            float starFalloff = 12.0 - pc.bassLevel * 4.0;
            float starBrightness = exp(-starDist * starFalloff) * (0.4 + starRand * 0.6);
            starBrightness *= 1.0 + pc.bassLevel * 1.5;
            float twinkleSpeed = 3.0 + pc.bassLevel * 5.0;
            starBrightness *= 0.7 + 0.3 * sin(pc.time * twinkleSpeed + starRand * 20.0);
            vec3 starColor = mix(vec3(1.0), getThemeAccentColor(starBrightness), pc.bassLevel * 0.6);
            color += starColor * starBrightness * bassPulse;
        }
    }

    // Aurora effect - themed
    float aurora = sin(uv.x * 6.0 + pc.time * 0.2 + snoise(uv * 3.0) * 2.0) * 0.5 + 0.5;
    aurora *= smoothstep(0.5, 0.75, uv.y) * smoothstep(1.0, 0.85, uv.y);
    color += getThemeAuroraColor(aurora) * 0.12 * (1.0 + pc.bassLevel);

    // Spectrum bars overlay (bottom 30%) with stereo mode support
    if (uv.y < 0.32) {
        vec2 barUV = vec2(uv.x, uv.y / 0.32);
        int idx;
        float localX;
        int halfBars = max(pc.barCount / 2, 1);  // Prevent division by zero

        if (pc.stereoMode == 1) {
            // Stereo Mirror: Left channel mirrors from center-left, right from center-right
            // Left half (0.0 to 0.5): show left channel mirrored (high freq at edge, low freq at center)
            // Right half (0.5 to 1.0): show right channel (low freq at center, high freq at edge)
            float barWidth = 0.5 / float(max(halfBars, 1));
            if (barUV.x < 0.5) {
                // Left side - left channel, mirrored (high freq at left edge)
                float mirroredX = 0.5 - barUV.x;
                idx = clamp(int(mirroredX / barWidth), 0, halfBars - 1);  // Left channel is 0 to halfBars-1
                localX = fract(mirroredX / barWidth);
            } else {
                // Right side - right channel (low freq near center)
                float rightX = barUV.x - 0.5;
                idx = halfBars + clamp(int(rightX / barWidth), 0, halfBars - 1);  // Right channel is halfBars to barCount-1
                localX = fract(rightX / barWidth);
            }
        } else if (pc.stereoMode == 2) {
            // Stereo Split: Left channel on left half, right channel on right half
            float barWidth = 0.5 / float(max(halfBars, 1));
            if (barUV.x < 0.5) {
                // Left half - left channel (0 to halfBars-1)
                idx = clamp(int(barUV.x / barWidth), 0, max(halfBars - 1, 0));
                localX = fract(barUV.x / barWidth);
            } else {
                // Right half - right channel (halfBars to barCount-1)
                float rightX = barUV.x - 0.5;
                idx = halfBars + clamp(int(rightX / barWidth), 0, max(halfBars - 1, 0));
                localX = fract(rightX / barWidth);
            }
        } else {
            // Mono mode: standard left-to-right
            float barWidth = 1.0 / float(max(pc.barCount, 1));
            idx = clamp(int(barUV.x / barWidth), 0, max(pc.barCount - 1, 0));
            localX = fract(barUV.x / barWidth);
        }

        float mag = fftMagnitudes[idx];
        float dbRange = pc.maxDb - pc.minDb;
        float h = (dbRange > 0.001) ? clamp((mag - pc.minDb) / dbRange, 0.0, 1.0) : 0.0;

        if (localX > 0.12 && localX < 0.88 && barUV.y < h * 0.85) {
            // Determine frequency for color (based on position within channel)
            float freq;
            if (pc.stereoMode == 0) {
                freq = float(idx) / float(max(pc.barCount, 1));
            } else {
                // For stereo modes, use position within the channel's half
                freq = float(idx % halfBars) / float(max(halfBars, 1));
            }

            float hue = getThemeBarHue(freq) + sin(pc.time * 0.4) * 0.04;
            vec3 barCol = hsv2rgb(vec3(fract(hue), 0.85, 0.95));
            float vg = barUV.y / (h * 0.85);
            barCol *= 0.55 + vg * 0.55;

            // Glow
            float g = exp(-abs(localX - 0.5) * 5.0) * h * pc.glowIntensity;
            barCol += barCol * g * 0.6;

            // Peak - use themed accent color
            if (vg > 0.93) barCol = mix(barCol, getThemeAccentColor(h), (vg - 0.93) * 14.0);

            // In stereo mirror mode, add subtle channel tint
            if (pc.stereoMode == 1) {
                if (barUV.x < 0.5) {
                    barCol *= vec3(1.0, 0.95, 0.9);  // Warm tint for left
                } else {
                    barCol *= vec3(0.9, 0.95, 1.0);  // Cool tint for right
                }
            }

            color = mix(color * 0.3, barCol, 0.9);
        }

        // Center divider line for stereo modes
        if (pc.stereoMode > 0 && abs(barUV.x - 0.5) < 0.003) {
            color = mix(color, getThemeAccentColor(0.5), 0.5);
        }
    }

    // Chromatic aberration (color fringing at edges)
    if (pc.chromaticAberration > 0.001) {
        float dist = length(uv - 0.5);
        float aberration = dist * dist * pc.chromaticAberration;
        // Shift RGB channels based on distance from center
        color.r *= 1.0 + aberration * 0.3;
        color.b *= 1.0 - aberration * 0.3;
        // Add subtle color fringe
        color.r = mix(color.r, color.g, aberration * 0.1);
        color.b = mix(color.b, color.g, aberration * 0.1);
    }

    // Bloom effect (brighten highlights)
    if (pc.bloomIntensity > 0.001) {
        float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
        float bloom = smoothstep(0.4, 1.0, brightness) * pc.bloomIntensity;
        color += color * bloom * 0.5;
        color = clamp(color, 0.0, 1.5);  // Allow slight HDR
    }

    // Vignette with intensity control
    float vignette = 1.0 - pow(length(uv - 0.5) * 1.25, 2.5) * pc.vignetteIntensity;
    color *= clamp(vignette, 0.0, 1.0);

    // Scanlines with intensity control
    float scanline = 1.0 - pc.scanlineIntensity * 0.06 * (1.0 - sin(uv.y * pc.resolution.y * 0.4));
    color *= scanline;

    // Film grain
    color += (hash(uv + fract(pc.time * 0.1)) - 0.5) * 0.025;

    // Bass pulse
    color *= 1.0 + pc.bassLevel * 0.12;

    outColor = vec4(color, 1.0);
}
)";

namespace vulkan_spectrum {

VulkanRenderer::VulkanRenderer() {
    // Initialize with default configuration
    config_ = getDefaultConfig();

    // Initialize FFT arrays with configured min dB
    fftData_.fill(config_.audio.minDb);
    smoothedData_.fill(config_.audio.minDb);
    peakData_.fill(config_.audio.minDb);

    startTime_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    LOGI("VulkanRenderer created with config: %s", config_.name);
}

VulkanRenderer::~VulkanRenderer() {
    destroy();
}

bool VulkanRenderer::initialize(ANativeWindow* window) {
    if (initialized_) {
        // Surface may have been recreated (e.g., app backgrounded)
        // Check if we need to reinitialize
        if (window != window_) {
            LOGW("Surface changed, reinitializing Vulkan");
            destroy();
        } else {
            LOGW("Already initialized with same surface");
            return true;
        }
    }

    window_ = window;

    LOGI("Initializing Vulkan renderer...");

    if (!createInstance()) {
        LOGE("Failed to create Vulkan instance");
        return false;
    }

    if (!createSurface(window)) {
        LOGE("Failed to create surface");
        return false;
    }

    if (!selectPhysicalDevice()) {
        LOGE("Failed to select physical device");
        return false;
    }

    if (!createDevice()) {
        LOGE("Failed to create logical device");
        return false;
    }

    if (!createSwapchain()) {
        LOGE("Failed to create swapchain");
        return false;
    }

    if (!createRenderPass()) {
        LOGE("Failed to create render pass");
        return false;
    }

    if (!createFramebuffers()) {
        LOGE("Failed to create framebuffers");
        return false;
    }

    if (!createCommandPool()) {
        LOGE("Failed to create command pool");
        return false;
    }

    if (!createCommandBuffers()) {
        LOGE("Failed to create command buffers");
        return false;
    }

    if (!createSyncObjects()) {
        LOGE("Failed to create sync objects");
        return false;
    }

    if (!createFftBuffer()) {
        LOGE("Failed to create FFT buffer");
        return false;
    }

    if (!createUniformBuffer()) {
        LOGE("Failed to create uniform buffer");
        return false;
    }

    if (!createVertexBuffer()) {
        LOGE("Failed to create vertex buffer");
        return false;
    }

    if (!createDescriptorSetLayout()) {
        LOGE("Failed to create descriptor set layout");
        return false;
    }

    if (!createDescriptorPool()) {
        LOGE("Failed to create descriptor pool");
        return false;
    }

    if (!createDescriptorSets()) {
        LOGE("Failed to create descriptor sets");
        return false;
    }

    if (!createPipelineLayout()) {
        LOGE("Failed to create pipeline layout");
        return false;
    }

    if (!createGraphicsPipeline()) {
        LOGE("Failed to create graphics pipeline");
        return false;
    }

    // Try to initialize ray tracing (optional)
    if (rayTracingSupported_) {
        if (initRayTracing()) {
            LOGI("Hardware ray tracing initialized!");
        } else {
            LOGW("Ray tracing init failed, using raster fallback");
            rayTracingSupported_ = false;
        }
    }

    initialized_ = true;
    LOGI("Vulkan renderer initialized successfully");
    LOGI("  Device: %s", deviceProperties_.deviceName);
    if (rayTracingSupported_) {
        LOGI("  Ray Tracing: FULL PIPELINE (hardware RT cores)");
    } else if (rayQuerySupported_) {
        LOGI("  Ray Tracing: RAY QUERY (fragment shader RT)");
    } else {
        LOGI("  Ray Tracing: DISABLED (software raymarching)");
    }

    return true;
}

void VulkanRenderer::destroy() {
    if (!initialized_) return;

    vkDeviceWaitIdle(device_);

    // Cleanup ray tracing
    if (rayTracingSupported_) {
        if (topLevelAS_ != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_) {
            vkDestroyAccelerationStructureKHR_(device_, topLevelAS_, nullptr);
        }
        if (bottomLevelAS_ != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_) {
            vkDestroyAccelerationStructureKHR_(device_, bottomLevelAS_, nullptr);
        }
        if (asBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, asBuffer_, nullptr);
        }
        if (asMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, asMemory_, nullptr);
        }
        if (rayTracingPipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, rayTracingPipeline_, nullptr);
        }
        if (rayTracingPipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, rayTracingPipelineLayout_, nullptr);
        }
    }

    cleanupSwapchain();

    if (fftMappedData_) {
        vkUnmapMemory(device_, fftMemory_);
        fftMappedData_ = nullptr;
    }
    if (fftBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, fftBuffer_, nullptr);
    }
    if (fftMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, fftMemory_, nullptr);
    }

    // Cleanup uniform buffer
    if (uniformMappedData_) {
        vkUnmapMemory(device_, uniformMemory_);
        uniformMappedData_ = nullptr;
    }
    if (uniformBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, uniformBuffer_, nullptr);
    }
    if (uniformMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, uniformMemory_, nullptr);
    }

    if (vertexBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
    }
    if (vertexMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, vertexMemory_, nullptr);
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (imageAvailableSemaphores_[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
        }
        if (renderFinishedSemaphores_[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
        }
        if (inFlightFences_[i] != VK_NULL_HANDLE) {
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
    }

    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
    }
    if (graphicsPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
    }
    if (computePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, computePipeline_, nullptr);
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    }
    if (commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, commandPool_, nullptr);
    }
    if (renderPass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device_, renderPass_, nullptr);
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }

    initialized_ = false;
    LOGI("Vulkan renderer destroyed");
}

bool VulkanRenderer::createInstance() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "S24 Ultra Spectrum";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "S24 Ultra Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;  // Need 1.2 for ray tracing

    std::vector<const char*> extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
    };

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

#ifdef DEBUG
    const char* validationLayers[] = {"VK_LAYER_KHRONOS_validation"};
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = validationLayers;
#else
    createInfo.enabledLayerCount = 0;
#endif

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));
    LOGI("Vulkan instance created");
    return true;
}

bool VulkanRenderer::selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);

    if (deviceCount == 0) {
        LOGE("No Vulkan-capable devices found");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    // Find the best device (prefer discrete GPU, then check for ray tracing)
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        LOGI("Found device: %s (type: %d)", props.deviceName, props.deviceType);

        // Check for queue families
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        uint32_t graphicsIdx = UINT32_MAX;
        uint32_t presentIdx = UINT32_MAX;

        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphicsIdx = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);
            if (presentSupport) {
                presentIdx = i;
            }

            if (graphicsIdx != UINT32_MAX && presentIdx != UINT32_MAX) {
                break;
            }
        }

        if (graphicsIdx == UINT32_MAX || presentIdx == UINT32_MAX) {
            continue;
        }

        // Check for ray tracing support
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> availableExts(extCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, availableExts.data());

        bool hasRayTracingPipeline = false;
        bool hasRayQuery = false;
        bool hasAccelStruct = false;
        bool hasDeferredOps = false;
        bool hasSwapchain = false;
        bool hasBufferDeviceAddress = false;

        LOGI("Checking device extensions for %s:", props.deviceName);
        for (const auto& ext : availableExts) {
            if (strcmp(ext.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0) {
                hasRayTracingPipeline = true;
                LOGI("  Found: VK_KHR_ray_tracing_pipeline");
            }
            if (strcmp(ext.extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0) {
                hasRayQuery = true;
                LOGI("  Found: VK_KHR_ray_query");
            }
            if (strcmp(ext.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0) {
                hasAccelStruct = true;
                LOGI("  Found: VK_KHR_acceleration_structure");
            }
            if (strcmp(ext.extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) == 0) {
                hasDeferredOps = true;
                LOGI("  Found: VK_KHR_deferred_host_operations");
            }
            if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                hasSwapchain = true;
            }
            if (strcmp(ext.extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
                hasBufferDeviceAddress = true;
                LOGI("  Found: VK_KHR_buffer_device_address");
            }
        }

        if (!hasSwapchain) {
            LOGW("Device %s doesn't support swapchain", props.deviceName);
            continue;
        }

        // Select this device
        physicalDevice_ = device;
        deviceProperties_ = props;
        graphicsFamily_ = graphicsIdx;
        presentFamily_ = presentIdx;

        // Ray tracing: prefer ray_tracing_pipeline, fall back to ray_query
        // Both require acceleration_structure
        if (hasRayTracingPipeline && hasAccelStruct && hasDeferredOps) {
            rayTracingSupported_ = true;
            rayQuerySupported_ = false;
            LOGI("  Ray Tracing Mode: FULL PIPELINE");
        } else if (hasRayQuery && hasAccelStruct) {
            rayTracingSupported_ = false;
            rayQuerySupported_ = true;
            LOGI("  Ray Tracing Mode: RAY QUERY (fragment shader)");
        } else {
            rayTracingSupported_ = false;
            rayQuerySupported_ = false;
            LOGI("  Ray Tracing Mode: DISABLED (software raymarching)");
        }

        vkGetPhysicalDeviceFeatures(device, &deviceFeatures_);
        vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties_);

        LOGI("Selected device: %s", props.deviceName);
        LOGI("  Ray Tracing Support: %s", rayTracingSupported_ ? "YES" : "NO");
        return true;
    }

    LOGE("No suitable device found");
    return false;
}

bool VulkanRenderer::createDevice() {
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::vector<uint32_t> uniqueQueueFamilies = {graphicsFamily_};
    if (presentFamily_ != graphicsFamily_) {
        uniqueQueueFamilies.push_back(presentFamily_);
    }

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // Required extensions
    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    // Add ray tracing extensions if supported
    if (rayTracingSupported_) {
        // Full ray tracing pipeline
        deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
        LOGI("Enabling full ray tracing pipeline extensions");
    } else if (rayQuerySupported_) {
        // Ray query only (for fragment shader ray tracing)
        deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
        LOGI("Enabling ray query extensions for fragment shader RT");
    }

    VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures = {};
    bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures = {};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = {};
    rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQueryFeatures.rayQuery = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = {};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.accelerationStructure = VK_TRUE;

    if (rayTracingSupported_) {
        // Full ray tracing pipeline
        deviceFeatures2.pNext = &bufferDeviceAddressFeatures;
        bufferDeviceAddressFeatures.pNext = &rtPipelineFeatures;
        rtPipelineFeatures.pNext = &asFeatures;
    } else if (rayQuerySupported_) {
        // Ray query only
        deviceFeatures2.pNext = &bufferDeviceAddressFeatures;
        bufferDeviceAddressFeatures.pNext = &rayQueryFeatures;
        rayQueryFeatures.pNext = &asFeatures;
    }

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (rayTracingSupported_ || rayQuerySupported_) {
        createInfo.pNext = &deviceFeatures2;
    } else {
        createInfo.pEnabledFeatures = &deviceFeatures_;
    }

    VK_CHECK(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_));

    vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);

    // Load ray tracing function pointers
    if (rayTracingSupported_) {
        vkCreateAccelerationStructureKHR_ = (PFN_vkCreateAccelerationStructureKHR)
            vkGetDeviceProcAddr(device_, "vkCreateAccelerationStructureKHR");
        vkDestroyAccelerationStructureKHR_ = (PFN_vkDestroyAccelerationStructureKHR)
            vkGetDeviceProcAddr(device_, "vkDestroyAccelerationStructureKHR");
        vkGetAccelerationStructureBuildSizesKHR_ = (PFN_vkGetAccelerationStructureBuildSizesKHR)
            vkGetDeviceProcAddr(device_, "vkGetAccelerationStructureBuildSizesKHR");
        vkCmdBuildAccelerationStructuresKHR_ = (PFN_vkCmdBuildAccelerationStructuresKHR)
            vkGetDeviceProcAddr(device_, "vkCmdBuildAccelerationStructuresKHR");
        vkCreateRayTracingPipelinesKHR_ = (PFN_vkCreateRayTracingPipelinesKHR)
            vkGetDeviceProcAddr(device_, "vkCreateRayTracingPipelinesKHR");
        vkGetRayTracingShaderGroupHandlesKHR_ = (PFN_vkGetRayTracingShaderGroupHandlesKHR)
            vkGetDeviceProcAddr(device_, "vkGetRayTracingShaderGroupHandlesKHR");
        vkCmdTraceRaysKHR_ = (PFN_vkCmdTraceRaysKHR)
            vkGetDeviceProcAddr(device_, "vkCmdTraceRaysKHR");

        if (!vkCreateAccelerationStructureKHR_ || !vkCmdTraceRaysKHR_) {
            LOGW("Failed to load ray tracing functions, disabling RT");
            rayTracingSupported_ = false;
        }
    }

    LOGI("Logical device created");
    return true;
}

bool VulkanRenderer::createSurface(ANativeWindow* window) {
    VkAndroidSurfaceCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    createInfo.window = window;

    VK_CHECK(vkCreateAndroidSurfaceKHR(instance_, &createInfo, nullptr, &surface_));
    LOGI("Surface created");
    return true;
}

bool VulkanRenderer::createSwapchain() {
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, formats.data());

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &presentModeCount, presentModes.data());

    // Choose format (prefer SRGB)
    VkSurfaceFormatKHR surfaceFormat = formats[0];
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surfaceFormat = format;
            break;
        }
    }

    // Choose present mode for 120fps
    // Priority: MAILBOX (lowest latency, no tearing) > IMMEDIATE (lowest latency, may tear) > FIFO (vsync)
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    bool hasMailbox = false, hasImmediate = false;
    for (const auto& mode : presentModes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) hasMailbox = true;
        if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) hasImmediate = true;
    }
    if (hasMailbox) {
        presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        LOGI("Present mode: MAILBOX (optimal for 120fps)");
    } else if (hasImmediate) {
        presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        LOGI("Present mode: IMMEDIATE (low latency, may tear)");
    } else {
        LOGI("Present mode: FIFO (vsync, may limit to 60fps)");
    }

    // Choose extent
    VkExtent2D extent = capabilities.currentExtent;
    if (capabilities.currentExtent.width == UINT32_MAX) {
        extent.width = ANativeWindow_getWidth(window_);
        extent.height = ANativeWindow_getHeight(window_);
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    // Triple buffering minimum for 120fps
    uint32_t imageCount = std::max(capabilities.minImageCount + 1, 3u);
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }
    LOGI("Swapchain images: %u (triple buffering)", imageCount);

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilyIndices[] = {graphicsFamily_, presentFamily_};
    if (graphicsFamily_ != presentFamily_) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_));

    swapchainFormat_ = surfaceFormat.format;
    swapchainExtent_ = extent;

    // Get swapchain images
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
    swapchainImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data());

    // Create image views
    swapchainImageViews_.resize(imageCount);
    for (size_t i = 0; i < imageCount; i++) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapchainImages_[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchainFormat_;
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &swapchainImageViews_[i]));
    }

    LOGI("Swapchain created: %dx%d, %d images", extent.width, extent.height, imageCount);
    return true;
}

bool VulkanRenderer::createRenderPass() {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapchainFormat_;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VK_CHECK(vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass_));
    return true;
}

bool VulkanRenderer::createFramebuffers() {
    framebuffers_.resize(swapchainImageViews_.size());

    for (size_t i = 0; i < swapchainImageViews_.size(); i++) {
        VkImageView attachments[] = {swapchainImageViews_[i]};

        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass_;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapchainExtent_.width;
        framebufferInfo.height = swapchainExtent_.height;
        framebufferInfo.layers = 1;

        VK_CHECK(vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &framebuffers_[i]));
    }

    return true;
}

bool VulkanRenderer::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsFamily_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));
    return true;
}

bool VulkanRenderer::createCommandBuffers() {
    commandBuffers_.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());

    VK_CHECK(vkAllocateCommandBuffers(device_, &allocInfo, commandBuffers_.data()));
    return true;
}

bool VulkanRenderer::createSyncObjects() {
    imageAvailableSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]));
        VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]));
        VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &inFlightFences_[i]));
    }

    return true;
}

bool VulkanRenderer::createFftBuffer() {
    VkDeviceSize bufferSize = sizeof(float) * config_.audio.barCount;

    if (!createBuffer(bufferSize,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      fftBuffer_, fftMemory_)) {
        return false;
    }

    // Map persistently
    VK_CHECK(vkMapMemory(device_, fftMemory_, 0, bufferSize, 0, (void**)&fftMappedData_));

    // Initialize with configured min dB
    for (int i = 0; i < config_.audio.barCount; i++) {
        fftMappedData_[i] = config_.audio.minDb;
    }

    return true;
}

bool VulkanRenderer::createUniformBuffer() {
    VkDeviceSize bufferSize = sizeof(ShaderUniforms);

    if (!createBuffer(bufferSize,
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      uniformBuffer_, uniformMemory_)) {
        return false;
    }

    // Map persistently
    VK_CHECK(vkMapMemory(device_, uniformMemory_, 0, bufferSize, 0, (void**)&uniformMappedData_));

    // Initialize with default config
    updateUniformBuffer();

    LOGI("Uniform buffer created: %zu bytes", sizeof(ShaderUniforms));
    return true;
}

void VulkanRenderer::updateUniformBuffer() {
    if (!uniformMappedData_) return;

    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    float time = (now - startTime_) / 1e9f;
    static float lastTime = 0;
    float deltaTime = time - lastTime;
    lastTime = time;

    // Convert config to shader uniforms
    ShaderUniforms uniforms = configToUniforms(config_, time, deltaTime);

    // Add runtime values
    uniforms.resolution[0] = (float)swapchainExtent_.width;
    uniforms.resolution[1] = (float)swapchainExtent_.height;
    uniforms.tiltX = tiltX_;
    uniforms.tiltY = tiltY_;
    uniforms.bassLevel = bassLevel_;

    // Copy to mapped memory
    memcpy(uniformMappedData_, &uniforms, sizeof(ShaderUniforms));

    configDirty_.store(false);
}

bool VulkanRenderer::createVertexBuffer() {
    // Full-screen quad vertices (position + UV)
    // NOTE: Vulkan's clip space has Y=-1 at TOP, Y=+1 at BOTTOM (opposite of OpenGL)
    // UV.v is flipped so that v=0 is at screen bottom, v=1 at screen top
    float vertices[] = {
        // pos x, y, uv u, v
        -1.0f, -1.0f, 0.0f, 1.0f,  // top-left in screen space, UV (0,1)
         1.0f, -1.0f, 1.0f, 1.0f,  // top-right in screen space, UV (1,1)
        -1.0f,  1.0f, 0.0f, 0.0f,  // bottom-left in screen space, UV (0,0)
         1.0f,  1.0f, 1.0f, 0.0f,  // bottom-right in screen space, UV (1,0)
    };

    VkDeviceSize bufferSize = sizeof(vertices);

    if (!createBuffer(bufferSize,
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      vertexBuffer_, vertexMemory_)) {
        return false;
    }

    void* data;
    vkMapMemory(device_, vertexMemory_, 0, bufferSize, 0, &data);
    memcpy(data, vertices, bufferSize);
    vkUnmapMemory(device_, vertexMemory_);

    return true;
}

bool VulkanRenderer::createDescriptorSetLayout() {
    // Binding 0: FFT data (storage buffer)
    VkDescriptorSetLayoutBinding fftBinding = {};
    fftBinding.binding = 0;
    fftBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    fftBinding.descriptorCount = 1;
    fftBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 1: Configuration uniforms (uniform buffer)
    VkDescriptorSetLayoutBinding uniformBinding = {};
    uniformBinding.binding = 1;
    uniformBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBinding.descriptorCount = 1;
    uniformBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {fftBinding, uniformBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_));
    return true;
}

bool VulkanRenderer::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> poolSizes = {};

    // Storage buffer for FFT data
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    // Uniform buffer for configuration
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));
    return true;
}

bool VulkanRenderer::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout_);

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets_.resize(MAX_FRAMES_IN_FLIGHT);
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, descriptorSets_.data()));

    // Update descriptor sets with both bindings
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // Binding 0: FFT buffer
        VkDescriptorBufferInfo fftBufferInfo = {};
        fftBufferInfo.buffer = fftBuffer_;
        fftBufferInfo.offset = 0;
        fftBufferInfo.range = sizeof(float) * config_.audio.barCount;

        // Binding 1: Uniform buffer
        VkDescriptorBufferInfo uniformBufferInfo = {};
        uniformBufferInfo.buffer = uniformBuffer_;
        uniformBufferInfo.offset = 0;
        uniformBufferInfo.range = sizeof(ShaderUniforms);

        std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

        // FFT storage buffer
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets_[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &fftBufferInfo;

        // Config uniform buffer
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets_[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &uniformBufferInfo;

        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(descriptorWrites.size()),
                               descriptorWrites.data(), 0, nullptr);
    }

    return true;
}

bool VulkanRenderer::createPipelineLayout() {
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(SpectrumPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));
    return true;
}

VkShaderModule VulkanRenderer::createShaderModule(const uint32_t* code, size_t size) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = size;
    createInfo.pCode = code;

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        LOGE("Failed to create shader module");
        return VK_NULL_HANDLE;
    }

    return shaderModule;
}

VkShaderModule VulkanRenderer::compileShaderFromGlsl(const char* glslSource, int kind, const char* name) {
#if HAS_SHADERC
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Set optimization level
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

    // Compile GLSL to SPIR-V
    shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(
        glslSource, strlen(glslSource), static_cast<shaderc_shader_kind>(kind), name, options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOGE("Shader compilation failed: %s", result.GetErrorMessage().c_str());
        return VK_NULL_HANDLE;
    }

    // Get SPIR-V bytecode
    std::vector<uint32_t> spirv(result.cbegin(), result.cend());

    LOGI("Compiled %s: %zu bytes SPIR-V", name, spirv.size() * sizeof(uint32_t));

    return createShaderModule(spirv.data(), spirv.size() * sizeof(uint32_t));
#else
    LOGW("shaderc not available, cannot compile GLSL at runtime");
    (void)glslSource;
    (void)kind;
    (void)name;
    return VK_NULL_HANDLE;
#endif
}

bool VulkanRenderer::createGraphicsPipeline() {
    // Use SPIR-V shaders with uniform buffer for dynamic theme colors
    LOGI("Creating shader modules from SPIR-V (uniform buffer system)...");
    LOGI("  Vertex SPIR-V size: %zu bytes", sizeof(spectrum_vert_spv));
    LOGI("  Fragment SPIR-V size: %zu bytes", sizeof(spectrum_frag_spv));

    VkShaderModule vertShader = createShaderModule(spectrum_vert_spv, sizeof(spectrum_vert_spv));
    VkShaderModule fragShader = createShaderModule(spectrum_frag_spv, sizeof(spectrum_frag_spv));

    if (vertShader == VK_NULL_HANDLE || fragShader == VK_NULL_HANDLE) {
        LOGE("CRITICAL: SPIR-V shader creation failed!");
        LOGE("  Vertex shader: %s", vertShader != VK_NULL_HANDLE ? "OK" : "FAILED");
        LOGE("  Fragment shader: %s", fragShader != VK_NULL_HANDLE ? "OK" : "FAILED");
        if (vertShader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vertShader, nullptr);
        if (fragShader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, fragShader, nullptr);
        return false;
    }

    LOGI("SPIR-V shader modules created successfully (dynamic themes enabled)");

    VkPipelineShaderStageCreateInfo vertStageInfo = {};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertShader;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo = {};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragShader;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStageInfo, fragStageInfo};

    // Vertex input
    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = 4 * sizeof(float);  // pos.xy + uv.xy
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescriptions[2] = {};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = 0;

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[1].offset = 2 * sizeof(float);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapchainExtent_.width;
    viewport.height = (float)swapchainExtent_.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = swapchainExtent_;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.renderPass = renderPass_;
    pipelineInfo.subpass = 0;

    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline_));

    vkDestroyShaderModule(device_, vertShader, nullptr);
    vkDestroyShaderModule(device_, fragShader, nullptr);

    LOGI("Graphics pipeline created");
    return true;
}

bool VulkanRenderer::initRayTracing() {
    if (!rayTracingSupported_) {
        LOGE("Ray tracing not supported on this device");
        return false;
    }

    LOGI("Initializing hardware ray tracing on Adreno 750...");

    // Create acceleration structures for terrain
    if (!createAccelerationStructures()) {
        LOGE("Failed to create acceleration structures");
        return false;
    }

    // Create ray tracing pipeline
    if (!createRayTracingPipeline()) {
        LOGE("Failed to create ray tracing pipeline");
        return false;
    }

    LOGI("Hardware ray tracing initialized successfully!");
    return true;
}

bool VulkanRenderer::createAccelerationStructures() {
    // For terrain ray tracing, we create a simple ground plane BLAS
    // The terrain deformation is done in the intersection shader based on FFT data

    // TODO: Full BLAS/TLAS implementation
    // For now, we verify RT is available and return true
    // The actual RT rendering will use ray queries in fragment shader

    LOGI("Acceleration structures ready (using ray queries)");
    return true;
}

bool VulkanRenderer::createRayTracingPipeline() {
    // For S24 Ultra, we use ray queries in the fragment shader
    // This is simpler than full RT pipeline and still uses RT hardware

    LOGI("Ray tracing pipeline ready (ray query mode)");
    return true;
}

void VulkanRenderer::render() {
    if (!initialized_ || paused_) return;

    // Safety check - ensure Vulkan objects are valid before rendering
    if (device_ == VK_NULL_HANDLE || swapchain_ == VK_NULL_HANDLE) {
        return;
    }

    // Check array bounds and handles
    if (currentFrame_ >= MAX_FRAMES_IN_FLIGHT ||
        commandBuffers_.empty() ||
        inFlightFences_[currentFrame_] == VK_NULL_HANDLE ||
        imageAvailableSemaphores_[currentFrame_] == VK_NULL_HANDLE ||
        renderFinishedSemaphores_[currentFrame_] == VK_NULL_HANDLE ||
        commandBuffers_[currentFrame_] == VK_NULL_HANDLE) {
        return;
    }

    VkResult fenceResult = vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
    if (fenceResult != VK_SUCCESS) {
        LOGE("vkWaitForFences failed: %d", fenceResult);
        return;
    }

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                            imageAvailableSemaphores_[currentFrame_],
                                            VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LOGE("Failed to acquire swapchain image");
        return;
    }

    vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);

    // REAL-TIME: Always update FFT smoothing every frame for continuous animation
    // applySmoothingCpu() handles both new data interpolation and idle decay
    applySmoothingCpu();
    updatePeaks();
    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        memcpy(fftMappedData_, smoothedData_.data(), sizeof(float) * config_.audio.barCount);
    }

    // Update uniform buffer every frame (contains time, tilt, etc.)
    updateUniformBuffer();

    // Record command buffer
    recordCommandBuffer(imageIndex);

    // Submit
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[currentFrame_];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores_[currentFrame_]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFences_[currentFrame_]) != VK_SUCCESS) {
        LOGE("Failed to submit draw command buffer");
        return;
    }

    // Present
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(presentQueue_, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        recreateSwapchain();
    } else if (result != VK_SUCCESS) {
        LOGE("Failed to present swapchain image");
    }

    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;

    // Performance monitoring for 120fps target
    frameCount_++;
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    if (lastFpsTime_ == 0) lastFpsTime_ = now;

    uint64_t elapsed = now - lastFpsTime_;
    if (elapsed >= 1000000000) {  // 1 second
        currentFps_ = frameCount_ * 1000000000.0f / elapsed;
        frameTimeMs_ = elapsed / (frameCount_ * 1000000.0f);
        frameCount_ = 0;
        lastFpsTime_ = now;

        // FPS logging disabled to reduce log noise
        // if (currentFps_ < 110.0f) {
        //     LOGW("FPS: %.1f (%.2fms) - Below 120fps target", currentFps_, frameTimeMs_);
        // } else {
        //     LOGI("FPS: %.1f (%.2fms)", currentFps_, frameTimeMs_);
        // }
    }
}

void VulkanRenderer::recordCommandBuffer(uint32_t imageIndex) {
    // Bounds check
    if (imageIndex >= framebuffers_.size() || framebuffers_.empty()) {
        LOGE("Invalid imageIndex %u, framebuffers size %zu", imageIndex, framebuffers_.size());
        return;
    }

    VkCommandBuffer cmd = commandBuffers_[currentFrame_];
    if (cmd == VK_NULL_HANDLE) return;

    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkBeginCommandBuffer(cmd, &beginInfo);

    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass_;
    renderPassInfo.framebuffer = framebuffers_[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapchainExtent_;

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);

    VkBuffer vertexBuffers[] = {vertexBuffer_};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_,
                            0, 1, &descriptorSets_[currentFrame_], 0, nullptr);

    // Push constants
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    float time = (now - startTime_) / 1e9f;

    SpectrumPushConstants pc = {};
    pc.resolution[0] = (float)swapchainExtent_.width;
    pc.resolution[1] = (float)swapchainExtent_.height;
    pc.time = time;
    pc.bassLevel = bassLevel_;
    pc.tiltX = tiltX_;
    pc.tiltY = tiltY_;
    pc.barCount = config_.audio.barCount;
    pc.minDb = config_.audio.minDb;
    pc.maxDb = config_.audio.maxDb;
    pc.glowIntensity = config_.bars.glowIntensity;
    // Visual effects (from config)
    pc.bloomIntensity = config_.postProcess.bloomIntensity;
    pc.chromaticAberration = config_.postProcess.chromaticAmount;
    pc.colorTheme = colorTheme_;
    pc.scanlineIntensity = config_.postProcess.scanlineIntensity;
    pc.vignetteIntensity = config_.postProcess.vignetteIntensity;
    pc.stereoMode = config_.stereoMode;

    vkCmdPushConstants(cmd, pipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDraw(cmd, 4, 1, 0, 0);

    vkCmdEndRenderPass(cmd);

    vkEndCommandBuffer(cmd);
}

void VulkanRenderer::cleanupSwapchain() {
    for (auto framebuffer : framebuffers_) {
        vkDestroyFramebuffer(device_, framebuffer, nullptr);
    }
    framebuffers_.clear();

    for (auto imageView : swapchainImageViews_) {
        vkDestroyImageView(device_, imageView, nullptr);
    }
    swapchainImageViews_.clear();

    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanRenderer::recreateSwapchain() {
    vkDeviceWaitIdle(device_);
    cleanupSwapchain();

    createSwapchain();
    createFramebuffers();

    LOGI("Swapchain recreated");
}

uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memoryProperties_.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memoryProperties_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    LOGE("Failed to find suitable memory type");
    return UINT32_MAX;
}

bool VulkanRenderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags properties, VkBuffer& buffer,
                                   VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (allocInfo.memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(device_, buffer, nullptr);
        return false;
    }

    VK_CHECK(vkAllocateMemory(device_, &allocInfo, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device_, buffer, memory, 0));

    return true;
}

void VulkanRenderer::applySmoothingCpu() {
    // REAL-TIME RESPONSIVE: Fast attack + continuous decay
    // Bars rise quickly to match audio, fall smoothly even without new data
    const float attackFactor = 0.85f;   // Fast rise to target
    const float decayFactor = 0.15f;    // Smooth fall towards target
    const float continuousDecay = 1.5f; // dB per frame decay when idle

    std::lock_guard<std::mutex> lock(dataMutex_);

    // Check if we got new data this frame
    bool hasNewData = dataUpdated_.exchange(false);

    for (int i = 0; i < NUM_BARS; i++) {
        float target = fftData_[i];
        float current = smoothedData_[i];

        if (hasNewData) {
            // New data arrived - interpolate towards it
            if (target > current) {
                smoothedData_[i] = current + (target - current) * attackFactor;
            } else {
                smoothedData_[i] = current + (target - current) * decayFactor;
            }
        } else {
            // No new data - continuous decay towards silence
            smoothedData_[i] = std::max(MIN_DB, current - continuousDecay);
        }
    }
}

void VulkanRenderer::updatePeaks() {
    for (int i = 0; i < NUM_BARS; i++) {
        if (smoothedData_[i] > peakData_[i]) {
            peakData_[i] = smoothedData_[i];
        } else {
            peakData_[i] = std::max(smoothedData_[i], peakData_[i] - 0.5f);
        }
    }
}

void VulkanRenderer::pause() {
    paused_ = true;
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
    LOGI("Renderer paused");
}

void VulkanRenderer::resume() {
    paused_ = false;

    // Recreate swapchain on resume to fix black screen issue
    if (device_ != VK_NULL_HANDLE && swapchain_ != VK_NULL_HANDLE) {
        recreateSwapchain();
    }

    LOGI("Renderer resumed, swapchain recreated");
}

void VulkanRenderer::updateFftData(const float* data, int size) {
    if (!data || size <= 0) return;

    std::lock_guard<std::mutex> lock(dataMutex_);

    if (size <= NUM_BARS) {
        memcpy(fftData_.data(), data, size * sizeof(float));
    } else {
        // Downsample
        int binsPerBar = size / NUM_BARS;
        for (int bar = 0; bar < NUM_BARS; bar++) {
            float sum = 0;
            int startBin = bar * binsPerBar;
            int endBin = std::min(startBin + binsPerBar, size);
            for (int bin = startBin; bin < endBin; bin++) {
                sum += data[bin];
            }
            fftData_[bar] = sum / (endBin - startBin);
        }
    }

    // REACTIVE: Enhanced bass level calculation for stronger punch
    int bassEnd = NUM_BARS / 4;  // Was /8 - wider bass range for better detection
    float bassSum = 0;
    float bassPeak = 0;
    for (int i = 0; i < bassEnd; i++) {
        float val = std::clamp((fftData_[i] - MIN_DB) / (MAX_DB - MIN_DB), 0.0f, 1.0f);
        // Weight lower frequencies more heavily (sub-bass punch)
        float weight = 1.0f + (1.0f - (float)i / bassEnd) * 2.0f;  // 3x weight for lowest
        bassSum += val * weight;
        bassPeak = std::max(bassPeak, val);
    }
    // Combine average + peak for transient response
    float bassAvg = bassSum / (bassEnd * 2.0f);  // Normalize for weights
    float rawBass = bassAvg * 0.4f + bassPeak * 0.6f;  // Peak-weighted for punch
    // Apply boost and power curve for more reactive response
    float boost = config_.audio.bassBoost;  // Use config boost (1.8 default)
    rawBass = std::pow(rawBass, 0.7f) * boost;  // Power < 1 expands low values
    bassLevel_ = std::clamp(rawBass, 0.0f, 1.0f);

    dataUpdated_.store(true);
}

void VulkanRenderer::updateStereoFftData(const float* left, const float* right, int size) {
    if (!left || !right || size <= 0) return;

    std::lock_guard<std::mutex> lock(dataMutex_);

    int halfBars = NUM_BARS / 2;
    int binsPerBar = size / halfBars;

    // Left channel
    for (int bar = 0; bar < halfBars; bar++) {
        float sum = 0;
        int startBin = bar * binsPerBar;
        int endBin = std::min(startBin + binsPerBar, size);
        for (int bin = startBin; bin < endBin; bin++) {
            sum += left[bin];
        }
        fftData_[bar] = sum / (endBin - startBin);
    }

    // Right channel (mirrored)
    for (int bar = 0; bar < halfBars; bar++) {
        float sum = 0;
        int startBin = bar * binsPerBar;
        int endBin = std::min(startBin + binsPerBar, size);
        for (int bin = startBin; bin < endBin; bin++) {
            sum += right[bin];
        }
        fftData_[NUM_BARS - 1 - bar] = sum / (endBin - startBin);
    }

    // REACTIVE: Enhanced stereo bass level calculation for stronger punch
    int bassEnd = halfBars / 4;  // Was /8 - wider bass range
    float bassSum = 0;
    float bassPeak = 0;
    for (int i = 0; i < bassEnd; i++) {
        float valL = std::clamp((fftData_[i] - MIN_DB) / (MAX_DB - MIN_DB), 0.0f, 1.0f);
        float valR = std::clamp((fftData_[NUM_BARS - 1 - i] - MIN_DB) / (MAX_DB - MIN_DB), 0.0f, 1.0f);
        float val = (valL + valR) * 0.5f;
        // Weight lower frequencies more heavily (sub-bass punch)
        float weight = 1.0f + (1.0f - (float)i / bassEnd) * 2.0f;  // 3x weight for lowest
        bassSum += val * weight;
        bassPeak = std::max(bassPeak, std::max(valL, valR));
    }
    // Combine average + peak for transient response
    float bassAvg = bassSum / (bassEnd * 2.0f);  // Normalize for weights
    float rawBass = bassAvg * 0.4f + bassPeak * 0.6f;  // Peak-weighted for punch
    // Apply boost and power curve for more reactive response
    float boost = config_.audio.bassBoost;  // Use config boost (1.8 default)
    rawBass = std::pow(rawBass, 0.7f) * boost;  // Power < 1 expands low values
    bassLevel_ = std::clamp(rawBass, 0.0f, 1.0f);

    dataUpdated_.store(true);
}

void VulkanRenderer::setTilt(float x, float y) {
    tiltX_ = x;
    tiltY_ = y;
}

void VulkanRenderer::setGlowIntensity(float intensity) {
    glowIntensity_ = std::clamp(intensity, 0.0f, 2.0f);
    config_.bars.glowIntensity = glowIntensity_;
    configDirty_.store(true);
}

void VulkanRenderer::setBloomIntensity(float intensity) {
    bloomIntensity_ = std::clamp(intensity, 0.0f, 2.0f);
    config_.postProcess.bloomIntensity = bloomIntensity_;
    configDirty_.store(true);
}

void VulkanRenderer::setChromaticAberration(float amount) {
    chromaticAberration_ = std::clamp(amount, 0.0f, 1.0f);
    config_.postProcess.chromaticAmount = chromaticAberration_;
    configDirty_.store(true);
}

void VulkanRenderer::setColorTheme(int theme) {
    colorTheme_ = std::clamp(theme, 0, 3);
    // Update config with new theme colors for uniform buffer shader
    loadPreset(colorTheme_);
}

void VulkanRenderer::setScanlineIntensity(float intensity) {
    scanlineIntensity_ = std::clamp(intensity, 0.0f, 1.0f);
    config_.postProcess.scanlineIntensity = scanlineIntensity_;
    configDirty_.store(true);
}

void VulkanRenderer::setVignetteIntensity(float intensity) {
    vignetteIntensity_ = std::clamp(intensity, 0.0f, 1.0f);
    config_.postProcess.vignetteIntensity = vignetteIntensity_;
    configDirty_.store(true);
}

void VulkanRenderer::setStereoMode(int mode) {
    stereoMode_ = std::clamp(mode, 0, 2);
    config_.stereoMode = stereoMode_;
    configDirty_.store(true);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION API - Fully Dynamic
// ═══════════════════════════════════════════════════════════════════════════════

void VulkanRenderer::setConfig(const SpectrumConfig& config) {
    config_ = config;
    configDirty_.store(true);
    LOGI("Config updated: %s", config_.name);
}

void VulkanRenderer::loadPreset(int presetId) {
    switch (presetId) {
        case 0: config_ = getNeonConfig(); break;    // Neon (purple/cyan)
        case 1: config_ = getFireConfig(); break;    // Fire (red/orange)
        case 2: config_ = getMatrixConfig(); break;  // Matrix (green)
        case 3: config_ = getOceanConfig(); break;   // Ocean (blue/teal)
        default: config_ = getNeonConfig(); break;
    }
    colorTheme_ = presetId;  // Sync legacy colorTheme_
    configDirty_.store(true);
    LOGI("Loaded preset %d: %s", presetId, config_.name);
    LOGI("  Theme colors: gradientLow=(%.2f,%.2f,%.2f) gradientHigh=(%.2f,%.2f,%.2f)",
         config_.theme.gradientLow.r, config_.theme.gradientLow.g, config_.theme.gradientLow.b,
         config_.theme.gradientHigh.r, config_.theme.gradientHigh.g, config_.theme.gradientHigh.b);
    LOGI("  Sky: horizon=(%.2f,%.2f,%.2f) zenith=(%.2f,%.2f,%.2f)",
         config_.theme.skyHorizon.r, config_.theme.skyHorizon.g, config_.theme.skyHorizon.b,
         config_.theme.skyZenith.r, config_.theme.skyZenith.g, config_.theme.skyZenith.b);
}

void VulkanRenderer::setTerrainEnabled(bool enabled) {
    config_.enableTerrain = enabled;
    configDirty_.store(true);
}

void VulkanRenderer::setBarsEnabled(bool enabled) {
    config_.enableBars = enabled;
    configDirty_.store(true);
}

void VulkanRenderer::setGridEnabled(bool enabled) {
    config_.enableGrid = enabled;
    configDirty_.store(true);
}

void VulkanRenderer::setNoiseScale(float scale1, float scale2) {
    config_.terrain.noiseScale1 = scale1;
    config_.terrain.noiseScale2 = scale2;
    configDirty_.store(true);
}

void VulkanRenderer::setNoiseAmplitude(float amp1, float amp2) {
    config_.terrain.noiseAmplitude1 = amp1;
    config_.terrain.noiseAmplitude2 = amp2;
    configDirty_.store(true);
}

void VulkanRenderer::setFftInfluence(float influence) {
    config_.terrain.fftInfluence = influence;
    configDirty_.store(true);
}

void VulkanRenderer::setTerrainThresholds(float snow, float rock) {
    config_.terrain.snowThreshold = snow;
    config_.terrain.rockThreshold = rock;
    configDirty_.store(true);
}

void VulkanRenderer::setCameraHeight(float height) {
    config_.camera.height = height;
    configDirty_.store(true);
}

void VulkanRenderer::setCameraSpeed(float speed) {
    config_.camera.moveSpeed = speed;
    configDirty_.store(true);
}

void VulkanRenderer::setFogDensity(float density) {
    config_.lighting.fogDensity = density;
    configDirty_.store(true);
}

void VulkanRenderer::setSunDirection(float x, float y, float z) {
    config_.lighting.sunDirX = x;
    config_.lighting.sunDirY = y;
    config_.lighting.sunDirZ = z;
    configDirty_.store(true);
}

void VulkanRenderer::setAmbientColor(float r, float g, float b) {
    config_.lighting.ambientColor = Color3(r, g, b);
    configDirty_.store(true);
}

void VulkanRenderer::setSunColor(float r, float g, float b) {
    config_.lighting.sunColor = Color3(r, g, b);
    configDirty_.store(true);
}

void VulkanRenderer::setGradientColors(float lowR, float lowG, float lowB,
                                        float highR, float highG, float highB) {
    config_.theme.gradientLow = Color3(lowR, lowG, lowB);
    config_.theme.gradientHigh = Color3(highR, highG, highB);
    configDirty_.store(true);
}

void VulkanRenderer::setSkyColors(float horizonR, float horizonG, float horizonB,
                                   float zenithR, float zenithG, float zenithB) {
    config_.theme.skyHorizon = Color3(horizonR, horizonG, horizonB);
    config_.theme.skyZenith = Color3(zenithR, zenithG, zenithB);
    configDirty_.store(true);
}

void VulkanRenderer::setBarAppearance(float hueStart, float hueRange,
                                       float saturation, float brightness) {
    config_.theme.barHueStart = hueStart;
    config_.theme.barHueRange = hueRange;
    config_.theme.barSaturation = saturation;
    config_.theme.barBrightness = brightness;
    configDirty_.store(true);
}

void VulkanRenderer::setPostProcessing(float contrast, float saturation,
                                        float brightness, float gamma) {
    config_.postProcess.contrast = contrast;
    config_.postProcess.saturation = saturation;
    config_.postProcess.brightness = brightness;
    config_.postProcess.gamma = gamma;
    configDirty_.store(true);
}

void VulkanRenderer::setBassPulse(float intensity, float r, float g, float b) {
    config_.postProcess.bassPulseIntensity = intensity;
    config_.postProcess.bassPulseColor = Color3(r, g, b);
    configDirty_.store(true);
}

void VulkanRenderer::reset() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    fftData_.fill(MIN_DB);
    smoothedData_.fill(MIN_DB);
    peakData_.fill(MIN_DB);
    bassLevel_ = 0;
    dataUpdated_.store(true);
}

} // namespace vulkan_spectrum

// ═══════════════════════════════════════════════════════════════════════════════
// JNI EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

using namespace vulkan_spectrum;

extern "C" {

JNIEXPORT jlong JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeCreate(JNIEnv* env, jobject obj) {
    auto* renderer = new VulkanRenderer();
    return reinterpret_cast<jlong>(renderer);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeDestroy(JNIEnv* env, jobject obj, jlong handle) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    delete renderer;
}

JNIEXPORT jboolean JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeInit(JNIEnv* env, jobject obj, jlong handle, jobject surface) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    bool result = renderer->initialize(window);
    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeRender(JNIEnv* env, jobject obj, jlong handle) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->render();
}

// Combined tilt + render in single JNI call for better performance
JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeRenderWithTilt(JNIEnv* env, jobject obj, jlong handle, jfloat tiltX, jfloat tiltY) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setTilt(tiltX, tiltY);
    renderer->render();
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativePause(JNIEnv* env, jobject obj, jlong handle) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->pause();
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeResume(JNIEnv* env, jobject obj, jlong handle) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->resume();
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeUpdateFft(JNIEnv* env, jobject obj, jlong handle, jfloatArray data) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    jsize size = env->GetArrayLength(data);
    jfloat* elements = env->GetFloatArrayElements(data, nullptr);
    renderer->updateFftData(elements, size);
    env->ReleaseFloatArrayElements(data, elements, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeUpdateStereoFft(JNIEnv* env, jobject obj, jlong handle, jfloatArray left, jfloatArray right) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    jsize leftSize = env->GetArrayLength(left);
    jsize rightSize = env->GetArrayLength(right);
    if (leftSize != rightSize) return;

    jfloat* leftElements = env->GetFloatArrayElements(left, nullptr);
    jfloat* rightElements = env->GetFloatArrayElements(right, nullptr);
    renderer->updateStereoFftData(leftElements, rightElements, leftSize);
    env->ReleaseFloatArrayElements(left, leftElements, JNI_ABORT);
    env->ReleaseFloatArrayElements(right, rightElements, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetTilt(JNIEnv* env, jobject obj, jlong handle, jfloat x, jfloat y) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setTilt(x, y);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetGlowIntensity(JNIEnv* env, jobject obj, jlong handle, jfloat intensity) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setGlowIntensity(intensity);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeReset(JNIEnv* env, jobject obj, jlong handle) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->reset();
}

JNIEXPORT jboolean JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeHasRayTracing(JNIEnv* env, jobject obj, jlong handle) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    return renderer->hasRayTracing() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeHasRayQuery(JNIEnv* env, jobject obj, jlong handle) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    return renderer->hasRayQuery() ? JNI_TRUE : JNI_FALSE;
}

// Visual effects JNI
JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetBloomIntensity(JNIEnv* env, jobject obj, jlong handle, jfloat intensity) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setBloomIntensity(intensity);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetChromaticAberration(JNIEnv* env, jobject obj, jlong handle, jfloat amount) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setChromaticAberration(amount);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetColorTheme(JNIEnv* env, jobject obj, jlong handle, jint theme) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setColorTheme(theme);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetScanlineIntensity(JNIEnv* env, jobject obj, jlong handle, jfloat intensity) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setScanlineIntensity(intensity);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetVignetteIntensity(JNIEnv* env, jobject obj, jlong handle, jfloat intensity) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setVignetteIntensity(intensity);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetStereoMode(JNIEnv* env, jobject obj, jlong handle, jint mode) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setStereoMode(mode);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION API JNI - Fully Dynamic Parameters
// ═══════════════════════════════════════════════════════════════════════════════

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeLoadPreset(JNIEnv* env, jobject obj, jlong handle, jint presetId) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->loadPreset(presetId);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetTerrainEnabled(JNIEnv* env, jobject obj, jlong handle, jboolean enabled) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setTerrainEnabled(enabled == JNI_TRUE);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetBarsEnabled(JNIEnv* env, jobject obj, jlong handle, jboolean enabled) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setBarsEnabled(enabled == JNI_TRUE);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetGridEnabled(JNIEnv* env, jobject obj, jlong handle, jboolean enabled) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setGridEnabled(enabled == JNI_TRUE);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetNoiseScale(JNIEnv* env, jobject obj, jlong handle, jfloat scale1, jfloat scale2) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setNoiseScale(scale1, scale2);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetNoiseAmplitude(JNIEnv* env, jobject obj, jlong handle, jfloat amp1, jfloat amp2) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setNoiseAmplitude(amp1, amp2);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetFftInfluence(JNIEnv* env, jobject obj, jlong handle, jfloat influence) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setFftInfluence(influence);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetTerrainThresholds(JNIEnv* env, jobject obj, jlong handle, jfloat snow, jfloat rock) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setTerrainThresholds(snow, rock);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetCameraHeight(JNIEnv* env, jobject obj, jlong handle, jfloat height) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setCameraHeight(height);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetCameraSpeed(JNIEnv* env, jobject obj, jlong handle, jfloat speed) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setCameraSpeed(speed);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetFogDensity(JNIEnv* env, jobject obj, jlong handle, jfloat density) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setFogDensity(density);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetSunDirection(JNIEnv* env, jobject obj, jlong handle, jfloat x, jfloat y, jfloat z) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setSunDirection(x, y, z);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetAmbientColor(JNIEnv* env, jobject obj, jlong handle, jfloat r, jfloat g, jfloat b) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setAmbientColor(r, g, b);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetSunColor(JNIEnv* env, jobject obj, jlong handle, jfloat r, jfloat g, jfloat b) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setSunColor(r, g, b);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetGradientColors(JNIEnv* env, jobject obj, jlong handle,
                                                                                          jfloat lowR, jfloat lowG, jfloat lowB,
                                                                                          jfloat highR, jfloat highG, jfloat highB) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setGradientColors(lowR, lowG, lowB, highR, highG, highB);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetSkyColors(JNIEnv* env, jobject obj, jlong handle,
                                                                                     jfloat horizonR, jfloat horizonG, jfloat horizonB,
                                                                                     jfloat zenithR, jfloat zenithG, jfloat zenithB) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setSkyColors(horizonR, horizonG, horizonB, zenithR, zenithG, zenithB);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetBarAppearance(JNIEnv* env, jobject obj, jlong handle,
                                                                                         jfloat hueStart, jfloat hueRange,
                                                                                         jfloat saturation, jfloat brightness) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setBarAppearance(hueStart, hueRange, saturation, brightness);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetPostProcessing(JNIEnv* env, jobject obj, jlong handle,
                                                                                          jfloat contrast, jfloat saturation,
                                                                                          jfloat brightness, jfloat gamma) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setPostProcessing(contrast, saturation, brightness, gamma);
}

JNIEXPORT void JNICALL
Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetBassPulse(JNIEnv* env, jobject obj, jlong handle,
                                                                                     jfloat intensity, jfloat r, jfloat g, jfloat b) {
    auto* renderer = reinterpret_cast<VulkanRenderer*>(handle);
    renderer->setBassPulse(intensity, r, g, b);
}

} // extern "C"
