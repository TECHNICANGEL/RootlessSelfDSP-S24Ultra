#!/bin/bash
# S24 Ultra - Vulkan Shader Compilation Script
# Compiles GLSL shaders to SPIR-V and generates C headers
#
# Requirements: glslangValidator (from Vulkan SDK or Android SDK)
#
# Usage: ./compile_shaders.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Find shader compiler (glslc from NDK or glslangValidator from Vulkan SDK)
COMPILER=""
USE_GLSLC=false

# Try glslc from Android NDK first (supports -fshader-stage flag)
if command -v glslc &> /dev/null; then
    COMPILER="glslc"
    USE_GLSLC=true
elif [ -n "$ANDROID_SDK_ROOT" ]; then
    # Find newest NDK with shader-tools
    NDK_GLSLC=$(find "$ANDROID_SDK_ROOT/ndk" -name "glslc*" -path "*shader-tools*" 2>/dev/null | sort -V | tail -1)
    if [ -n "$NDK_GLSLC" ]; then
        COMPILER="$NDK_GLSLC"
        USE_GLSLC=true
    fi
elif [ -d "/mnt/c/Users" ]; then
    # WSL: Look in Windows Android SDK
    NDK_GLSLC=$(find /mnt/c/Users/*/AppData/Local/Android/Sdk/ndk -name "glslc.exe" -path "*shader-tools*" 2>/dev/null | sort -V | tail -1)
    if [ -n "$NDK_GLSLC" ]; then
        COMPILER="$NDK_GLSLC"
        USE_GLSLC=true
    fi
fi

# Fall back to glslangValidator
if [ -z "$COMPILER" ]; then
    if command -v glslangValidator &> /dev/null; then
        COMPILER="glslangValidator"
    elif [ -n "$VULKAN_SDK" ]; then
        if [ -f "$VULKAN_SDK/bin/glslangValidator" ]; then
            COMPILER="$VULKAN_SDK/bin/glslangValidator"
        elif [ -f "$VULKAN_SDK/Bin/glslangValidator.exe" ]; then
            COMPILER="$VULKAN_SDK/Bin/glslangValidator.exe"
        fi
    fi
fi

if [ -z "$COMPILER" ]; then
    echo "ERROR: No shader compiler found!"
    echo "Options:"
    echo "  1. Install Android NDK (includes glslc in shader-tools)"
    echo "  2. Install Vulkan SDK from https://vulkan.lunarg.com/"
    echo "  3. Set VULKAN_SDK environment variable"
    exit 1
fi

echo "Using shader compiler: $COMPILER"
echo ""

# Function to compile shader and generate C header
compile_shader() {
    local shader_file=$1
    local base_name=$(basename "$shader_file" | sed 's/\./_/g')
    local spv_file="${shader_file}.spv"
    local header_name="${base_name}.h"
    local array_name="${base_name}_spv"

    echo "Compiling $shader_file..."

    # Compile to SPIR-V using appropriate compiler
    if [ "$USE_GLSLC" = true ]; then
        # Determine shader stage from extension
        local stage=""
        case "$shader_file" in
            *.vert) stage="vertex" ;;
            *.frag) stage="fragment" ;;
            *.comp) stage="compute" ;;
            *.geom) stage="geometry" ;;
            *.tesc) stage="tesscontrol" ;;
            *.tese) stage="tesseval" ;;
            *) echo "ERROR: Unknown shader type for $shader_file"; exit 1 ;;
        esac
        $COMPILER -fshader-stage=$stage "$shader_file" -o "$spv_file"
    else
        # glslangValidator
        $COMPILER -V "$shader_file" -o "$spv_file"
    fi

    if [ ! -f "$spv_file" ]; then
        echo "ERROR: Failed to compile $shader_file"
        exit 1
    fi

    echo "Generating $header_name..."

    # Generate C header with properly packed uint32_t values
    # SPIR-V is defined as a stream of 32-bit words (little-endian)
    {
        echo "/**"
        echo " * Auto-generated SPIR-V header"
        echo " * Source: $shader_file"
        echo " * Generated: $(date)"
        echo " */"
        echo ""
        echo "#ifndef ${base_name^^}_H"
        echo "#define ${base_name^^}_H"
        echo ""
        echo "#include <stdint.h>"
        echo ""
        echo "static const uint32_t ${array_name}[] = {"

        # Convert binary SPIR-V to properly packed uint32_t values (little-endian)
        # Read 4 bytes at a time and pack them into uint32_t
        hexdump -v -e '1/4 "    0x%08x,\n"' "$spv_file"

        echo "};"
        echo ""
        echo "#endif // ${base_name^^}_H"
    } > "$header_name"

    # Cleanup SPIR-V binary (we keep the header)
    rm -f "$spv_file"

    echo "  -> Generated $header_name ($(wc -c < "$header_name") bytes)"
    echo ""
}

# Compile shaders
if [ -f "spectrum.vert" ]; then
    compile_shader "spectrum.vert"
fi

if [ -f "spectrum.frag" ]; then
    compile_shader "spectrum.frag"
fi

if [ -f "terrain.comp" ]; then
    compile_shader "terrain.comp"
fi

echo "Done! Shader headers generated successfully."
echo ""
echo "Headers created:"
ls -la *.h 2>/dev/null || echo "No headers found"
