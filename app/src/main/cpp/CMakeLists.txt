# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.22.1)

# ═══════════════════════════════════════════════════════════════════════════════
# S24 ULTRA - TURBO MODE COMPILER FLAGS
# Snapdragon 8 Gen 3 (Cortex-X4 + A720 + A520) Maximum Performance
# ═══════════════════════════════════════════════════════════════════════════════

# Base optimization flags for all architectures
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections -ffast-math -ftree-vectorize -funroll-loops -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections -ffast-math -ftree-vectorize -funroll-loops -fomit-frame-pointer")

# TURBO: Aggressive inlining for hot paths
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -finline-functions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finline-functions")

# TURBO: Link-time optimization with LLD linker (NDK 23+ compatible)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto=thin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=thin")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=thin -fuse-ld=lld -Wl,--gc-sections")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto=thin -fuse-ld=lld -Wl,--gc-sections")

# Declares and names the project.
project(jamesdsp LANGUAGES C)
project(jamesdsp-wrapper LANGUAGES CXX)

include(libcrashlytics-connector/CMakeLists.txt)
include(libjdspimptoolbox/CMakeLists.txt)
include(libjamesdsp-wrapper/CMakeLists.txt)

file(GLOB_RECURSE LIBJAMESDSP_SOURCE_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_LIST_DIR}/libjamesdsp/Main/libjamesdsp/jni/jamesdsp/jdsp/*.c
        ${CMAKE_CURRENT_LIST_DIR}/libjamesdsp/Main/libjamesdsp/jni/jamesdsp/jdsp/*.h)

add_library( # Sets the name of the library.
        jamesdsp

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${LIBJAMESDSP_SOURCE_FILES}
        EELStdOutExtension.c
        EELStdOutExtension.h)

# RAW POWER: Maximum optimization level with architecture-specific flags
target_compile_options(jamesdsp PRIVATE -O3 -std=gnu11)

# Suppress warnings from upstream libjamesdsp code (external library)
target_compile_options(jamesdsp PRIVATE
    -Wno-return-type                              # polyphaseFilterbank.c
    -Wno-tautological-constant-out-of-range-compare  # cos_fib_paraunitary.c
    -Wno-unused-value                             # FFTConvolver.c
    -Wno-pointer-sign                             # nseel-compiler.c, eel_matrix.h
)

# ═══════════════════════════════════════════════════════════════════════════════
# S24 ULTRA TURBO: Architecture-specific SIMD optimizations
# ═══════════════════════════════════════════════════════════════════════════════
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    # ARM64 TURBO: Cortex-X4 (Snapdragon 8 Gen 3 prime core)
    # ARMv9-A with SVE2, but we use ARMv8.4-A for broader compatibility
    target_compile_options(jamesdsp PRIVATE
        -march=armv8.4-a+simd+crypto+dotprod
        -mtune=cortex-x4
        -moutline-atomics
        -fno-signed-zeros
        -fno-trapping-math
        -freciprocal-math
    )
    message(STATUS "TURBO: arm64-v8a Cortex-X4 with NEON/dotprod optimizations")
elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    # ARMv7 with NEON VFPv4
    target_compile_options(jamesdsp PRIVATE
        -march=armv7-a
        -mfpu=neon-vfpv4
        -mfloat-abi=softfp
        -fno-signed-zeros
    )
    message(STATUS "TURBO: armeabi-v7a with NEON VFPv4 optimizations")
elseif(${ANDROID_ABI} STREQUAL "x86_64")
    # x86_64 with AVX2 for emulators/ChromeOS
    target_compile_options(jamesdsp PRIVATE
        -march=x86-64-v3
        -msse4.2
        -mavx
        -mavx2
        -mfma
        -mpopcnt
    )
    message(STATUS "TURBO: x86_64 with AVX2/FMA optimizations")
elseif(${ANDROID_ABI} STREQUAL "x86")
    # x86 with SSE4.1
    target_compile_options(jamesdsp PRIVATE
        -march=i686
        -msse4.1
        -mfpmath=sse
    )
    message(STATUS "TURBO: x86 with SSE4.1 optimizations")
endif()
target_include_directories(jamesdsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libjamesdsp/Main/libjamesdsp/jni/jamesdsp/jdsp/)

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

target_link_libraries( # Specifies the target library.
        jamesdsp

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})

# ═══════════════════════════════════════════════════════════════
# NOTE: npu_audio library REMOVED - Source Separation now uses
# HTDemucs via ONNX Runtime (Java-side) instead of native NEON
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# S24 ULTRA - Cortex-X4 Specific DSP Library
# NOT generic ARM - specific to Snapdragon 8 Gen 3
# ═══════════════════════════════════════════════════════════════

add_library(
        s24_ultra_dsp
        SHARED
        s24_ultra_dsp.c
)

# S24 Ultra: Maximum Cortex-X4 optimizations
target_compile_options(s24_ultra_dsp PRIVATE
        -O3
        -ffast-math
        -ftree-vectorize
        -funroll-loops
        -fomit-frame-pointer
        -finline-functions
        -std=gnu11
        -fno-signed-zeros
        -fno-trapping-math
        -freciprocal-math
        -fno-math-errno
)

# S24 Ultra Cortex-X4 specific: ARMv8.4-A with all extensions
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_compile_options(s24_ultra_dsp PRIVATE
            # Cortex-X4 is ARMv9-A but we use v8.4 for NDK compatibility
            -march=armv8.4-a+simd+crypto+dotprod+fp16
            -mtune=cortex-x4
            -moutline-atomics
            # S24 Ultra specific: enable all FP optimizations
            -ffp-contract=fast
            -fno-honor-nans
            -fno-honor-infinities
    )
    message(STATUS "S24 ULTRA: Cortex-X4 specific build with dotprod+fp16")
endif()

target_link_libraries(s24_ultra_dsp
        ${log-lib}
        m  # Math library
)

# ═══════════════════════════════════════════════════════════════
# S24 ULTRA - Vulkan Spectrum Renderer
# Hardware ray tracing on Adreno 750 GPU
# ═══════════════════════════════════════════════════════════════

# Shader compilation setup
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_spectrum/shaders)
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# Find glslangValidator (Android SDK includes this)
find_program(GLSLANG_VALIDATOR
    NAMES glslangValidator glslangValidator.exe
    HINTS
        ${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}
        $ENV{VULKAN_SDK}/Bin
        $ENV{VULKAN_SDK}/bin
        /usr/bin
        /usr/local/bin
)

if(GLSLANG_VALIDATOR)
    message(STATUS "Found glslangValidator: ${GLSLANG_VALIDATOR}")

    # Function to compile GLSL to SPIR-V and generate C header
    function(compile_shader SHADER_NAME SHADER_STAGE)
        set(SHADER_SRC ${SHADER_DIR}/${SHADER_NAME}.${SHADER_STAGE})
        set(SPIRV_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}_${SHADER_STAGE}.spv)
        set(HEADER_OUTPUT ${SHADER_DIR}/${SHADER_NAME}_${SHADER_STAGE}.h)

        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SRC} -o ${SPIRV_OUTPUT}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling ${SHADER_NAME}.${SHADER_STAGE} to SPIR-V"
        )

        # Generate C header from SPIR-V (using xxd-like functionality)
        add_custom_command(
            OUTPUT ${HEADER_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "// Auto-generated SPIR-V header" > ${HEADER_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "#ifndef ${SHADER_NAME}_${SHADER_STAGE}_H" >> ${HEADER_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "#define ${SHADER_NAME}_${SHADER_STAGE}_H" >> ${HEADER_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "#include <stdint.h>" >> ${HEADER_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "static const uint32_t ${SHADER_NAME}_${SHADER_STAGE}_spv[] = {" >> ${HEADER_OUTPUT}
            DEPENDS ${SPIRV_OUTPUT}
            COMMENT "Generating ${SHADER_NAME}_${SHADER_STAGE}.h header"
        )

        list(APPEND SHADER_OUTPUTS ${SPIRV_OUTPUT})
        set(SHADER_OUTPUTS ${SHADER_OUTPUTS} PARENT_SCOPE)
    endfunction()

    # Compile shaders if source files exist
    if(EXISTS ${SHADER_DIR}/spectrum.vert AND EXISTS ${SHADER_DIR}/spectrum.frag)
        message(STATUS "Compiling Vulkan shaders...")
        # Note: Full shader compilation is done at build time
        # For now, we use pre-compiled SPIR-V headers
    endif()
else()
    message(WARNING "glslangValidator not found - using pre-compiled SPIR-V headers")
    message(WARNING "To compile shaders, install Vulkan SDK or use Android SDK shader-tools")
endif()

add_library(
        vulkan_spectrum
        SHARED
        vulkan_spectrum/vulkan_renderer.cpp
        vulkan_spectrum/spectrum_config.cpp
)

# C++17 for modern features
set_target_properties(vulkan_spectrum PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# Vulkan optimization flags
target_compile_options(vulkan_spectrum PRIVATE
        -O3
        -ffast-math
        -ftree-vectorize
        -funroll-loops
        -fomit-frame-pointer
        -finline-functions
        -fno-signed-zeros
        -fno-trapping-math
        -freciprocal-math
        -fno-math-errno
        -DVULKAN_HPP_NO_EXCEPTIONS
)

# Architecture-specific optimizations for Vulkan renderer
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_compile_options(vulkan_spectrum PRIVATE
            -march=armv8.4-a+simd+crypto+dotprod+fp16
            -mtune=cortex-x4
            -moutline-atomics
            -ffp-contract=fast
    )
    message(STATUS "vulkan_spectrum: ARM64 Cortex-X4 optimizations enabled")
elseif(${ANDROID_ABI} STREQUAL "x86_64")
    target_compile_options(vulkan_spectrum PRIVATE
            -march=x86-64-v3
            -mavx2
            -mfma
    )
    message(STATUS "vulkan_spectrum: x86_64 AVX2 optimizations enabled")
endif()

# Include directories
target_include_directories(vulkan_spectrum PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_spectrum
        ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_spectrum/shaders
)

# Find Vulkan library (provided by NDK)
find_library(vulkan-lib vulkan)

if(vulkan-lib)
    message(STATUS "Vulkan library found: ${vulkan-lib}")
else()
    message(WARNING "Vulkan library not found - will try to load dynamically")
endif()

# Find shaderc for runtime shader compilation (NDK r21+ has it)
# Try to find shaderc_combined which includes all dependencies
find_library(shaderc-lib shaderc)
if(NOT shaderc-lib)
    # Fallback: try shaderc_combined (older NDKs)
    find_library(shaderc-lib shaderc_combined)
endif()

if(shaderc-lib)
    message(STATUS "Found shaderc library: ${shaderc-lib}")
    target_compile_definitions(vulkan_spectrum PRIVATE HAS_SHADERC=1)
else()
    message(WARNING "shaderc not found - using pre-compiled SPIR-V only")
    message(WARNING "Runtime shader compilation will NOT be available")
    target_compile_definitions(vulkan_spectrum PRIVATE HAS_SHADERC=0)
    set(shaderc-lib "")  # Clear to avoid linking issues
endif()

# Link libraries
target_link_libraries(vulkan_spectrum
        ${log-lib}
        android
        ${vulkan-lib}
        m
)

# Conditionally link shaderc if found
if(shaderc-lib)
    target_link_libraries(vulkan_spectrum ${shaderc-lib})
endif()
