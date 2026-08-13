#!/usr/bin/env bash
# Local / container build helper that mirrors GitHub Actions:
#   cmake -G Ninja -DUSE_CUDA=ON (+ ccache launchers when available)
#   ninja
#   ./tests/dllib_tests
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
export CCACHE_DIR="${CCACHE_DIR:-/ccache}"

mkdir -p "${BUILD_DIR}" \
  results/predictions_torch \
  results/predictions_custom

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && ! grep -q "Ninja" "${BUILD_DIR}/CMakeCache.txt"; then
  echo "[dev.sh] Existing ${BUILD_DIR} cache is not Ninja; clearing it."
  rm -rf "${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
fi

CMAKE_ARGS=(
  -S "${ROOT}"
  -B "${BUILD_DIR}"
  -G Ninja
  -DUSE_CUDA=ON
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

if command -v ccache >/dev/null 2>&1; then
  echo "[dev.sh] ccache enabled (${CCACHE_DIR})"
  CMAKE_ARGS+=(
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
  )
else
  echo "[dev.sh] ccache not found; compiling without a compiler launcher"
fi

if [[ "${BUILD_TYPE}" == "Debug" ]]; then
  CMAKE_ARGS+=(-DUSE_SANITIZERS="${USE_SANITIZERS:-ON}")
fi

echo "[dev.sh] Configuring (${BUILD_TYPE}) in ${BUILD_DIR}"
cmake "${CMAKE_ARGS[@]}"

echo "[dev.sh] Building with Ninja"
if command -v ninja >/dev/null 2>&1; then
  ninja -C "${BUILD_DIR}"
else
  cmake --build "${BUILD_DIR}" --parallel
fi

echo "[dev.sh] Running ./tests/dllib_tests"
cd "${BUILD_DIR}"
if [[ -x ./tests/dllib_tests.exe ]]; then
  ./tests/dllib_tests.exe
elif [[ -x ./tests/dllib_tests ]]; then
  ./tests/dllib_tests
else
  echo "[dev.sh] Test executable not found: ${BUILD_DIR}/tests/dllib_tests" >&2
  exit 1
fi
