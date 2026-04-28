#!/bin/bash
set -e

# Build script for Qwen-3 inference engine
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_TYPE="${1:-Release}"
BUILD_DIR="$PROJECT_ROOT/build"

echo "Building Qwen-3 Inference Engine..."
echo "Build type: $BUILD_TYPE"
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DQWEN3_ENABLE_AVX2=ON \
    -DQWEN3_ENABLE_AVX512=ON \
    -DQWEN3_ENABLE_OPENMP=ON \
    "$PROJECT_ROOT"

# Build
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build completed successfully!"
echo "Executable: $BUILD_DIR/qwenium"

# Test basic functionality
echo "Testing basic functionality..."
if [ -x "./qwenium" ]; then
    ./qwenium --help
    echo "Basic test passed!"
else
    echo "ERROR: Executable not found or not executable"
    exit 1
fi