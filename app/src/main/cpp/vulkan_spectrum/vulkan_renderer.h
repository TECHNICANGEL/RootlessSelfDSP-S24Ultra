/**
 * S24 ULTRA - Vulkan Spectrum Renderer with Hardware Ray Tracing
 *
 * Adreno 750 GPU Features:
 * - VK_KHR_ray_tracing_pipeline (hardware RT cores)
 * - VK_KHR_acceleration_structure
 * - 16-wide wave execution
 * - 2MB L2 cache
 *
 * This renderer uses Vulkan for maximum performance on the S24 Ultra
 *
 * FULLY DYNAMIC: All parameters come from configuration - nothing hardcoded
 */

#ifndef VULKAN_RENDERER_H
#define VULKAN_RENDERER_H

#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>
#include <jni.h>
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cmath>

#include "spectrum_config.h"

#define LOG_TAG "VulkanSpectrum"
#define LOGI(...) ((void)0)  // Disabled to reduce log noise
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) ((void)0)  // Disabled to reduce log noise

// Default spectrum configuration (can be overridden at runtime)
#define DEFAULT_NUM_BARS 256
#define DEFAULT_MIN_DB -80.0f
#define DEFAULT_MAX_DB 0.0f

// Check Vulkan result and log error
#define VK_CHECK(x) do { \
    VkResult err = x; \
    if (err != VK_SUCCESS) { \
        LOGE("Vulkan error %d at %s:%d", err, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

#define VK_CHECK_VOID(x) do { \
    VkResult err = x; \
    if (err != VK_SUCCESS) { \
        LOGE("Vulkan error %d at %s:%d", err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

namespace vulkan_spectrum {

// Push constants for spectrum shader (max 128 bytes on mobile)
struct SpectrumPushConstants {
    float resolution[2];    // 8 bytes
    float time;             // 4 bytes
    float bassLevel;        // 4 bytes
    float tiltX;            // 4 bytes
    float tiltY;            // 4 bytes
    int barCount;           // 4 bytes
    float minDb;            // 4 bytes
    float maxDb;            // 4 bytes
    float glowIntensity;    // 4 bytes
    // Visual effects (40 + 24 = 64 bytes total)
    float bloomIntensity;   // 4 bytes
    float chromaticAberration; // 4 bytes
    int colorTheme;         // 4 bytes (0=neon, 1=fire, 2=matrix, 3=ocean)
    float scanlineIntensity; // 4 bytes
    float vignetteIntensity; // 4 bytes
    int stereoMode;         // 4 bytes (0=mono, 1=stereo mirror, 2=stereo split)
};

// Push constants for terrain ray tracing
struct TerrainPushConstants {
    float resolution[2];
    float time;
    float bassLevel;
    float tiltX;
    float tiltY;
    float cameraHeight;
    float fogDensity;
};

class VulkanRenderer {
public:
    VulkanRenderer();
    ~VulkanRenderer();

    // Lifecycle
    bool initialize(ANativeWindow* window);
    void destroy();
    bool isInitialized() const { return initialized_; }

    // Rendering
    void render();
    void pause();
    void resume();

    // Performance monitoring
    float getFps() const { return currentFps_; }
    float getFrameTimeMs() const { return frameTimeMs_; }

    // Data updates
    void updateFftData(const float* data, int size);
    void updateStereoFftData(const float* left, const float* right, int size);
    void setTilt(float x, float y);
    void setGlowIntensity(float intensity);
    void reset();

    // Visual effects (legacy API - maps to config)
    void setBloomIntensity(float intensity);
    void setChromaticAberration(float amount);
    void setColorTheme(int theme);  // 0=neon, 1=fire, 2=matrix, 3=ocean
    void setScanlineIntensity(float intensity);
    void setVignetteIntensity(float intensity);
    void setStereoMode(int mode);   // 0=mono, 1=stereo mirror, 2=stereo split

    // Configuration API - FULLY DYNAMIC
    void setConfig(const SpectrumConfig& config);
    void loadPreset(int presetId);  // 0=neon, 1=fire, 2=matrix, 3=ocean
    const SpectrumConfig& getConfig() const { return config_; }

    // Individual parameter setters (for real-time tweaking)
    void setTerrainEnabled(bool enabled);
    void setBarsEnabled(bool enabled);
    void setGridEnabled(bool enabled);
    void setNoiseScale(float scale1, float scale2);
    void setNoiseAmplitude(float amp1, float amp2);
    void setFftInfluence(float influence);
    void setTerrainThresholds(float snow, float rock);
    void setCameraHeight(float height);
    void setCameraSpeed(float speed);
    void setFogDensity(float density);
    void setSunDirection(float x, float y, float z);
    void setAmbientColor(float r, float g, float b);
    void setSunColor(float r, float g, float b);
    void setGradientColors(float lowR, float lowG, float lowB, float highR, float highG, float highB);
    void setSkyColors(float horizonR, float horizonG, float horizonB, float zenithR, float zenithG, float zenithB);
    void setBarAppearance(float hueStart, float hueRange, float saturation, float brightness);
    void setPostProcessing(float contrast, float saturation, float brightness, float gamma);
    void setBassPulse(float intensity, float r, float g, float b);

    // Ray tracing support
    bool hasRayTracing() const { return rayTracingSupported_ || rayQuerySupported_; }
    bool hasRayQuery() const { return rayQuerySupported_; }
    bool hasFullRayTracing() const { return rayTracingSupported_; }

private:
    // Initialization
    bool createInstance();
    bool selectPhysicalDevice();
    bool createDevice();
    bool createSurface(ANativeWindow* window);
    bool createSwapchain();
    bool createRenderPass();
    bool createFramebuffers();
    bool createCommandPool();
    bool createCommandBuffers();
    bool createSyncObjects();
    bool createDescriptorSetLayout();
    bool createDescriptorPool();
    bool createDescriptorSets();
    bool createPipelineLayout();
    bool createGraphicsPipeline();
    bool createComputePipeline();
    bool createFftBuffer();
    bool createVertexBuffer();
    bool createUniformBuffer();
    void updateUniformBuffer();

    // Ray tracing initialization (optional)
    bool initRayTracing();
    bool createAccelerationStructures();
    bool createRayTracingPipeline();

    // Shader loading
    VkShaderModule createShaderModule(const uint32_t* code, size_t size);
    VkShaderModule compileShaderFromGlsl(const char* glslSource, int kind, const char* name);

    // Command recording
    void recordCommandBuffer(uint32_t imageIndex);
    void recordRayTracingCommands(VkCommandBuffer cmd);
    void recordRasterCommands(VkCommandBuffer cmd, uint32_t imageIndex);

    // Cleanup
    void cleanupSwapchain();
    void recreateSwapchain();

    // Vulkan objects
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;
    VkPipeline computePipeline_ = VK_NULL_HANDLE;

    // Ray tracing objects
    VkPipeline rayTracingPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout rayTracingPipelineLayout_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR topLevelAS_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR bottomLevelAS_ = VK_NULL_HANDLE;
    VkBuffer asBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory asMemory_ = VK_NULL_HANDLE;

    // Function pointers for ray tracing extensions
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ = nullptr;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR_ = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR_ = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR_ = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR_ = nullptr;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR_ = nullptr;

    // Swapchain images
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkDescriptorSet> descriptorSets_;
    VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_ = {0, 0};

    // Sync objects
    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    static constexpr int MAX_FRAMES_IN_FLIGHT = 3;  // Triple buffering for 120fps

    // FFT data buffer
    VkBuffer fftBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory fftMemory_ = VK_NULL_HANDLE;
    float* fftMappedData_ = nullptr;

    // Vertex buffer (for quad rendering)
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;

    // Uniform buffer (for all dynamic configuration)
    VkBuffer uniformBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory uniformMemory_ = VK_NULL_HANDLE;
    ShaderUniforms* uniformMappedData_ = nullptr;

    // Dynamic configuration
    SpectrumConfig config_;
    std::atomic<bool> configDirty_{true};

    // Render state
    std::array<float, DEFAULT_NUM_BARS> fftData_;
    std::array<float, DEFAULT_NUM_BARS> smoothedData_;
    std::array<float, DEFAULT_NUM_BARS> peakData_;
    std::mutex dataMutex_;
    std::atomic<bool> dataUpdated_{false};

    float bassLevel_ = 0.0f;
    float tiltX_ = 0.0f;
    float tiltY_ = 0.0f;
    float glowIntensity_ = 1.0f;
    uint64_t startTime_ = 0;

    // Visual effects state
    float bloomIntensity_ = 0.5f;
    float chromaticAberration_ = 0.0f;
    int colorTheme_ = 0;  // 0=neon, 1=fire, 2=matrix, 3=ocean
    float scanlineIntensity_ = 0.5f;
    float vignetteIntensity_ = 0.6f;
    int stereoMode_ = 0;  // 0=mono, 1=stereo mirror, 2=stereo split

    // State flags
    bool initialized_ = false;
    bool paused_ = false;
    bool rayTracingSupported_ = false;  // Full ray tracing pipeline
    bool rayQuerySupported_ = false;    // Ray query in fragment shader

    // Performance monitoring for 120fps target
    uint64_t frameCount_ = 0;
    uint64_t lastFpsTime_ = 0;
    float currentFps_ = 0.0f;
    float frameTimeMs_ = 0.0f;

    // Queue family indices
    uint32_t graphicsFamily_ = UINT32_MAX;
    uint32_t presentFamily_ = UINT32_MAX;

    // Physical device properties
    VkPhysicalDeviceProperties deviceProperties_;
    VkPhysicalDeviceFeatures deviceFeatures_;
    VkPhysicalDeviceMemoryProperties memoryProperties_;

    // Native window
    ANativeWindow* window_ = nullptr;

    // Helper functions
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& memory);
    void applySmoothingCpu();
    void updatePeaks();
};

} // namespace vulkan_spectrum

// JNI exports
extern "C" {
    JNIEXPORT jlong JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeCreate(JNIEnv* env, jobject obj);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeDestroy(JNIEnv* env, jobject obj, jlong handle);
    JNIEXPORT jboolean JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeInit(JNIEnv* env, jobject obj, jlong handle, jobject surface);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeRender(JNIEnv* env, jobject obj, jlong handle);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativePause(JNIEnv* env, jobject obj, jlong handle);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeResume(JNIEnv* env, jobject obj, jlong handle);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeUpdateFft(JNIEnv* env, jobject obj, jlong handle, jfloatArray data);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeUpdateStereoFft(JNIEnv* env, jobject obj, jlong handle, jfloatArray left, jfloatArray right);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetTilt(JNIEnv* env, jobject obj, jlong handle, jfloat x, jfloat y);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetGlowIntensity(JNIEnv* env, jobject obj, jlong handle, jfloat intensity);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeReset(JNIEnv* env, jobject obj, jlong handle);
    JNIEXPORT jboolean JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeHasRayTracing(JNIEnv* env, jobject obj, jlong handle);
    JNIEXPORT jboolean JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeHasRayQuery(JNIEnv* env, jobject obj, jlong handle);
    // Visual effects (legacy)
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetBloomIntensity(JNIEnv* env, jobject obj, jlong handle, jfloat intensity);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetChromaticAberration(JNIEnv* env, jobject obj, jlong handle, jfloat amount);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetColorTheme(JNIEnv* env, jobject obj, jlong handle, jint theme);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetScanlineIntensity(JNIEnv* env, jobject obj, jlong handle, jfloat intensity);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetVignetteIntensity(JNIEnv* env, jobject obj, jlong handle, jfloat intensity);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetStereoMode(JNIEnv* env, jobject obj, jlong handle, jint mode);

    // Configuration API - FULLY DYNAMIC
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeLoadPreset(JNIEnv* env, jobject obj, jlong handle, jint presetId);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetTerrainEnabled(JNIEnv* env, jobject obj, jlong handle, jboolean enabled);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetBarsEnabled(JNIEnv* env, jobject obj, jlong handle, jboolean enabled);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetGridEnabled(JNIEnv* env, jobject obj, jlong handle, jboolean enabled);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetNoiseScale(JNIEnv* env, jobject obj, jlong handle, jfloat scale1, jfloat scale2);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetNoiseAmplitude(JNIEnv* env, jobject obj, jlong handle, jfloat amp1, jfloat amp2);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetFftInfluence(JNIEnv* env, jobject obj, jlong handle, jfloat influence);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetTerrainThresholds(JNIEnv* env, jobject obj, jlong handle, jfloat snow, jfloat rock);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetCameraHeight(JNIEnv* env, jobject obj, jlong handle, jfloat height);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetCameraSpeed(JNIEnv* env, jobject obj, jlong handle, jfloat speed);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetFogDensity(JNIEnv* env, jobject obj, jlong handle, jfloat density);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetSunDirection(JNIEnv* env, jobject obj, jlong handle, jfloat x, jfloat y, jfloat z);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetAmbientColor(JNIEnv* env, jobject obj, jlong handle, jfloat r, jfloat g, jfloat b);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetSunColor(JNIEnv* env, jobject obj, jlong handle, jfloat r, jfloat g, jfloat b);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetGradientColors(JNIEnv* env, jobject obj, jlong handle, jfloat lowR, jfloat lowG, jfloat lowB, jfloat highR, jfloat highG, jfloat highB);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetSkyColors(JNIEnv* env, jobject obj, jlong handle, jfloat horizonR, jfloat horizonG, jfloat horizonB, jfloat zenithR, jfloat zenithG, jfloat zenithB);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetBarAppearance(JNIEnv* env, jobject obj, jlong handle, jfloat hueStart, jfloat hueRange, jfloat saturation, jfloat brightness);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetPostProcessing(JNIEnv* env, jobject obj, jlong handle, jfloat contrast, jfloat saturation, jfloat brightness, jfloat gamma);
    JNIEXPORT void JNICALL Java_me_timschneeberger_rootlessjamesdsp_view_VulkanSpectrumView_nativeSetBassPulse(JNIEnv* env, jobject obj, jlong handle, jfloat intensity, jfloat r, jfloat g, jfloat b);
}

#endif // VULKAN_RENDERER_H
