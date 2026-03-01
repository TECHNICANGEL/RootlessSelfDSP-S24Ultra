#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 fragUV;

// Push constants must match the fragment shader (SpectrumPushConstants in C++)
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
    // Visual effects
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
