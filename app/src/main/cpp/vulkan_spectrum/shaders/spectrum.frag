#version 450
#extension GL_EXT_control_flow_attributes : enable

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// ═══════════════════════════════════════════════════════════════════════════
// FFT DATA BUFFER
// ═══════════════════════════════════════════════════════════════════════════

layout(std430, binding = 0) readonly buffer FftData {
    float fftMagnitudes[];
};

// ═══════════════════════════════════════════════════════════════════════════
// UNIFORM BUFFER - ALL VALUES ARE DYNAMIC
// ═══════════════════════════════════════════════════════════════════════════

layout(std140, binding = 1) uniform Config {
    vec2 resolution;
    float time;
    float deltaTime;

    float tiltX;
    float tiltY;
    float cameraHeight;
    float cameraSpeed;

    float bassLevel;
    float midLevel;
    float highLevel;
    float overallLevel;

    int barCount;
    float minDb;
    float maxDb;
    float fftInfluence;

    vec4 gradientLow;
    vec4 gradientHigh;

    vec4 skyHorizon;
    vec4 skyZenith;

    vec4 terrainGrass;
    vec4 terrainRock;
    vec4 terrainSnow;

    float noiseScale1;
    float noiseScale2;
    float noiseAmp1;
    float noiseAmp2;

    float baseHeight;
    float snowThreshold;
    float rockThreshold;
    float gridThickness;

    vec3 sunDir;
    float sunIntensity;

    vec4 sunColor;
    vec4 ambientColor;

    float ambientIntensity;
    float fogDensity;
    float sunGlowPower;
    float sunGlowIntensity;

    float barWidthRatio;
    float barMaxHeight;
    float barMarginLeft;
    float barMarginRight;

    float barHueStart;
    float barHueRange;
    float barSaturation;
    float barBrightness;

    float vignetteIntensity;
    float vignetteRadius;
    float vignetteSoftness;
    float scanlineIntensity;

    float scanlineCount;
    float chromaticAmount;
    float bloomIntensity;
    float bloomThreshold;

    float contrast;
    float saturation;
    float brightness;
    float gamma;

    float bassPulseIntensity;
    float bassPulseR;
    float bassPulseG;
    float bassPulseB;

    int stereoMode;
    int enableTerrain;
    int enableBars;
    int enableGrid;

    int coarseSteps;
    int fineSteps;
    float maxDistance;
    float normalEpsilon;

    float glowIntensity;
    float fftGlowIntensity;
    float glowR;
    float glowG;

    float glowB;
    float _padding1;
    float _padding2;
    float _padding3;
};

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZED NOISE - Single instruction hash for 120fps
// ═══════════════════════════════════════════════════════════════════════════

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZED FFT SAMPLING - Branchless
// ═══════════════════════════════════════════════════════════════════════════

float sampleFFT(int bin) {
    float dbRange = maxDb - minDb;
    int idx = clamp(bin, 0, barCount - 1);
    float val = (fftMagnitudes[idx] - minDb) / max(dbRange, 0.001);
    return clamp(val, 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZED COLOR UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec3 getThemeColor(float intensity) {
    return mix(gradientLow.rgb, gradientHigh.rgb, intensity);
}

// ═══════════════════════════════════════════════════════════════════════════
// REALISTIC SKY - Atmospheric scattering + stars + clouds
// ═══════════════════════════════════════════════════════════════════════════

// Fractal noise for clouds (2 octaves for performance)
float fbm(vec2 p) {
    float f = noise(p) * 0.5;
    f += noise(p * 2.1) * 0.25;
    return f;
}

vec3 getSkyColor(vec3 rd) {
    float y = max(rd.y, 0.0);

    // Non-linear gradient for more realistic atmosphere
    float horizonBlend = 1.0 - pow(1.0 - y, 3.0);  // More horizon color near bottom
    float zenithBlend = pow(y, 0.7);  // Gradual transition to zenith

    // Base sky with atmospheric curve
    vec3 sky = mix(skyHorizon.rgb, skyZenith.rgb, zenithBlend);

    // Atmospheric haze near horizon (warmer, brighter)
    float haze = exp(-y * 4.0);
    vec3 hazeColor = mix(skyHorizon.rgb, sunColor.rgb * 0.3, 0.3);
    sky = mix(sky, hazeColor, haze * 0.4);

    // Sun with realistic glow layers
    float sunDot = max(dot(rd, sunDir), 0.0);

    // Inner sun core (sharp)
    float sunCore = smoothstep(0.995, 0.999, sunDot);
    sky += sunColor.rgb * sunCore * sunIntensity * 2.0;

    // Sun glow (medium)
    float sunGlow = pow(sunDot, 8.0);
    sky += sunColor.rgb * sunGlow * sunGlowIntensity * 0.5;

    // Atmospheric scattering (wide, soft)
    float scatter = pow(sunDot, 2.0);
    vec3 scatterColor = mix(sunColor.rgb, vec3(1.0, 0.7, 0.4), 0.5);
    sky += scatterColor * scatter * 0.15 * (1.0 - y);

    // Stars (only when looking up and sky is dark)
    float skyBrightness = dot(sky, vec3(0.299, 0.587, 0.114));
    if (y > 0.1 && skyBrightness < 0.4) {
        // Star field
        vec2 starUV = rd.xz / (rd.y + 0.001) * 20.0;
        float starHash = hash(floor(starUV));

        // Only some cells have stars
        if (starHash > 0.92) {
            vec2 starLocal = fract(starUV) - 0.5;
            vec2 starOffset = vec2(hash(floor(starUV) + 1.0), hash(floor(starUV) + 2.0)) - 0.5;
            starOffset *= 0.4;

            float starDist = length(starLocal - starOffset);
            float star = exp(-starDist * 25.0);

            // Twinkle
            float twinkle = 0.7 + 0.3 * sin(time * 3.0 + starHash * 50.0);
            star *= twinkle;

            // Star color (slight variation)
            vec3 starColor = mix(vec3(1.0), vec3(0.8, 0.9, 1.0), starHash);

            // Fade stars based on sky brightness
            float starFade = smoothstep(0.4, 0.1, skyBrightness) * smoothstep(0.1, 0.3, y);
            sky += starColor * star * starFade * 0.8;
        }
    }

    // Subtle cloud wisps (only in upper sky)
    if (y > 0.15) {
        vec2 cloudUV = rd.xz / (rd.y + 0.1) * 2.0 + time * 0.02;
        float cloud = fbm(cloudUV);
        cloud = smoothstep(0.3, 0.7, cloud);

        // Clouds lit by sun
        vec3 cloudColor = mix(skyHorizon.rgb * 1.2, sunColor.rgb * 0.8, sunDot * 0.5);
        float cloudFade = smoothstep(0.15, 0.4, y) * (1.0 - smoothstep(0.7, 1.0, y));
        sky = mix(sky, cloudColor, cloud * cloudFade * 0.25);
    }

    // Horizon glow (sunrise/sunset effect)
    float horizonGlow = exp(-abs(y) * 8.0);
    vec3 glowColor = mix(skyHorizon.rgb, sunColor.rgb * 0.6, 0.4);
    sky += glowColor * horizonGlow * 0.2;

    return sky;
}

// ═══════════════════════════════════════════════════════════════════════════
// TERRAIN - Peaks grow with distance, clear view
// ═══════════════════════════════════════════════════════════════════════════

float getTerrainHeight(vec2 pos) {
    // Two octaves of noise
    float h1 = noise(pos * noiseScale1) * noiseAmp1;
    float h2 = noise(pos * noiseScale2 + 7.3) * noiseAmp2;

    // Combine octaves
    float terrainBase = h1 * 0.7 + h2 * 0.3;

    // Base floor
    float h = max(terrainBase, baseHeight);

    // FFT boost - frequency-reactive peaks
    int bin = int(mod(pos.x * 2.5 + 50.0, float(barCount)));
    float fft = sampleFFT(bin);

    // FFT response with bass
    float fftBoost = fft * fftInfluence * (1.0 + bassLevel * 0.5);

    h += fftBoost;

    return h;
}

// Raymarching - balanced steps
float intersectTerrain(vec3 ro, vec3 rd) {
    float t = 0.5;

    // 8 steps for good balance of quality/performance
    [[unroll]] for (int i = 0; i < 8; i++) {
        vec3 p = ro + rd * t;
        float h = getTerrainHeight(p.xz);
        float diff = p.y - h;

        // Adaptive step
        float step = max(diff * 0.45, 0.12);
        t += (diff > 0.0) ? step : -step * 0.4;
    }

    return t;
}

// Fast normal using larger epsilon
vec3 getTerrainNormal(vec2 pos) {
    const float e = 0.2;  // Larger epsilon = faster, less accurate
    float h = getTerrainHeight(pos);
    float hx = getTerrainHeight(pos + vec2(e, 0.0));
    float hz = getTerrainHeight(pos + vec2(0.0, e));
    return normalize(vec3(h - hx, e, h - hz));
}

vec3 getTerrainMaterial(float h) {
    // Branchless material selection using smoothstep
    float snowFactor = smoothstep(snowThreshold - 0.5, snowThreshold + 0.5, h);
    float rockFactor = smoothstep(rockThreshold - 0.5, rockThreshold + 0.5, h);

    vec3 mat = mix(terrainGrass.rgb, terrainRock.rgb, rockFactor);
    return mix(mat, terrainSnow.rgb, snowFactor);
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZED SPECTRUM BARS
// ═══════════════════════════════════════════════════════════════════════════

vec4 drawBars(vec2 uv) {
    float invBarCount = 1.0 / float(max(barCount, 1));
    int barIndex = int(uv.x * float(barCount));
    barIndex = clamp(barIndex, 0, barCount - 1);

    float barLocalX = fract(uv.x * float(barCount));

    float magnitude = fftMagnitudes[barIndex];
    float dbRange = maxDb - minDb;
    float normalized = clamp((magnitude - minDb) / max(dbRange, 0.001), 0.0, 1.0);

    float barHeight = normalized * barMaxHeight;

    // Early exit - most pixels won't be in bars
    if (barLocalX < barMarginLeft || barLocalX > (1.0 - barMarginRight) || uv.y > barHeight) {
        return vec4(0.0);
    }

    float freq = float(barIndex) * invBarCount;
    float hue = fract(barHueStart + freq * barHueRange);

    vec3 barColor = hsv2rgb(vec3(hue, barSaturation, barBrightness));
    barColor *= 0.6 + (uv.y / max(barHeight, 0.001)) * 0.4;

    return vec4(barColor, 0.9);
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZED POST-PROCESSING - Conditional application
// ═══════════════════════════════════════════════════════════════════════════

vec3 applyPostProcess(vec3 color, vec2 uv) {
    // Vignette (only if enabled)
    if (vignetteIntensity > 0.01) {
        vec2 d = uv - 0.5;
        float vig = 1.0 - dot(d, d) * vignetteIntensity * 2.0;
        color *= max(vig, 0.0);
    }

    // Scanlines (only if enabled)
    if (scanlineIntensity > 0.01) {
        float scan = sin(uv.y * scanlineCount * 3.14159) * 0.5 + 0.5;
        color *= mix(1.0, scan, scanlineIntensity);
    }

    // Color grading (simplified)
    color *= brightness;
    color = (color - 0.5) * contrast + 0.5;

    // Fast gamma approximation
    color = max(color, vec3(0.0));
    color = color * inversesqrt(color + 0.1);  // Approximate gamma

    // Bass pulse
    color += vec3(bassPulseR, bassPulseG, bassPulseB) * bassLevel * bassPulseIntensity;

    return color;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN - REACTIVE MODE for 120fps on Adreno 750
// ═══════════════════════════════════════════════════════════════════════════

void main() {
    vec2 uv = fragUV;
    vec3 color;

    // REACTIVE: Bass-driven intensity multiplier
    float bassReact = bassLevel * bassLevel;  // Squared for more punch
    float reactPulse = 1.0 + bassReact * 0.5;

    // Fast path: Bars only (no raymarching)
    if (enableTerrain == 0) {
        if (enableBars != 0) {
            vec4 bars = drawBars(uv);
            // REACTIVE: Sky pulses with bass
            vec3 skyPulse = mix(skyHorizon.rgb, skyZenith.rgb, uv.y + bassReact * 0.2);
            skyPulse += getThemeColor(bassLevel) * bassReact * 0.3;
            color = mix(skyPulse, bars.rgb, bars.a);
        } else {
            color = mix(skyHorizon.rgb, skyZenith.rgb, uv.y);
            color += getThemeColor(bassLevel) * bassReact * 0.2;
        }
        color = applyPostProcess(color, uv);
        outColor = vec4(color, 1.0);
        return;
    }

    // Terrain path
    float aspect = resolution.x / resolution.y;

    // REACTIVE: Camera shake on bass hits
    vec2 shake = vec2(
        sin(time * 30.0) * bassReact * 0.02,
        cos(time * 25.0) * bassReact * 0.015
    );

    // REACTIVE: Zoom pulse on bass
    float zoomPulse = 1.0 - bassReact * 0.08;
    vec2 screenUV = (uv * 2.0 - 1.0) * vec2(aspect, 1.0) * zoomPulse + shake;

    // Camera setup with bass-reactive height bob
    float camZ = time * cameraSpeed;
    float heightBob = sin(time * 2.0) * 0.1 + bassReact * 0.3;
    vec3 ro = vec3(tiltX * 1.5, cameraHeight + tiltY * 0.3 + heightBob, camZ - 4.0);
    vec3 target = vec3(0.0, 1.0 + bassReact * 0.5, camZ + 6.0);

    vec3 forward = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    vec3 rd = normalize(forward + screenUV.x * right + screenUV.y * up);

    // Early sky exit for upward rays
    if (rd.y > 0.2) {
        color = getSkyColor(rd);
        // REACTIVE: Sky glow on bass
        color += getThemeColor(1.0) * bassReact * 0.15;
    } else {
        float t = intersectTerrain(ro, rd);

        if (t < maxDistance * 0.8) {
            vec3 p = ro + rd * t;
            vec3 n = getTerrainNormal(p.xz);
            float h = getTerrainHeight(p.xz);

            // Material
            vec3 albedo = getTerrainMaterial(h);

            // REACTIVE: Grid glows with bass and FFT
            if (enableGrid != 0) {
                vec2 grid = abs(fract(p.xz) - 0.5);
                float gridLine = step(gridThickness, max(grid.x, grid.y));

                // Grid becomes brighter and more colorful with bass
                float gridGlow = (1.0 - gridLine) * (0.3 + bassReact * 0.7);
                albedo *= 0.7 + gridLine * 0.3;
                albedo += getThemeColor(bassLevel) * gridGlow;
            }

            // Simple lighting
            float NdotL = max(dot(n, sunDir), 0.0);
            vec3 light = ambientColor.rgb * ambientIntensity + sunColor.rgb * NdotL * sunIntensity;
            color = albedo * light;

            // REACTIVE: FFT glow with pulsing intensity
            int bin = int(mod(p.x * 2.0 + 50.0, 32.0)) * barCount / 32;
            float fft = sampleFFT(bin);
            float glowPulse = fftGlowIntensity * reactPulse;
            color += getThemeColor(fft) * fft * glowPulse;

            // REACTIVE: Extra bloom on high FFT areas
            if (fft > 0.6) {
                color += getThemeColor(1.0) * (fft - 0.6) * 2.0 * bassReact;
            }

            // Fast fog - less fog when bass hits
            float dynamicFog = fogDensity * (1.0 - bassReact * 0.3);
            float fog = 1.0 - exp(-t * dynamicFog);
            color = mix(color, skyHorizon.rgb, fog);
        } else {
            color = getSkyColor(rd);
            color += getThemeColor(1.0) * bassReact * 0.1;
        }
    }

    // Overlay bars
    if (enableBars != 0) {
        vec4 bars = drawBars(uv);
        color = mix(color, bars.rgb, bars.a);
    }

    // REACTIVE: Boost saturation on bass hits
    float satBoost = 1.0 + bassReact * 0.3;
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(gray), color, satBoost);

    color = applyPostProcess(color, uv);
    outColor = vec4(color, 1.0);
}
