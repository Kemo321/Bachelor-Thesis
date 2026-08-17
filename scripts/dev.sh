#!/usr/bin/env bash
# Local / container build helper that mirrors GitHub Actions:
#   cmake -G Ninja -DUSE_CUDA=ON (+ ccache launchers when available)
#   ninja
#   ./tests/dllib_tests
#
# Windows: do not use WSL bash.exe from System32. Prefer:
#   powershell -File scripts/dev.ps1
# which loads vcvars64 and Git Bash.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "${MSYSTEM:-}" || "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ]]; then
  if [[ -z "${INCLUDE:-}" ]]; then
    echo "[dev.sh] MSVC INCLUDE is unset; cl.exe cannot find CRT headers (stdlib.h)." >&2
    echo "[dev.sh] On Windows run:  powershell -File scripts/dev.ps1" >&2
    echo "[dev.sh] Do not use WSL bash.exe from System32; that is not Git Bash." >&2
    exit 1
  fi
fi

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
export CCACHE_DIR="${CCACHE_DIR:-/ccache}"

# Container toolkits are often newer than the host driver. Emit real SASS for
# the attached GPU so CUB/Thrust do not JIT PTX the driver cannot load.
# shellcheck source=cuda_env.sh
source "${ROOT}/scripts/cuda_env.sh"
if [[ -n "${GPU_CC:-}" ]]; then
  echo "[dev.sh] GPU compute capability ${GPU_CC} -> CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi

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
if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}")
fi
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  CMAKE_ARGS+=(-DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}")
fi

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

echo "[dev.sh] Building dllib_tests with Ninja"
# Default matches CI Debug: unit tests only. Torch baseline binaries still
# compile against at::Tensor + YOLOLoss(dl::Tensor) and are not part of this script.
NINJA_TARGETS="${NINJA_TARGETS:-dllib_tests}"
# shellcheck disable=SC2086
if command -v ninja >/dev/null 2>&1; then
  ninja -C "${BUILD_DIR}" ${NINJA_TARGETS}
else
  cmake --build "${BUILD_DIR}" --parallel --target ${NINJA_TARGETS}
fi

echo "[dev.sh] Running ./tests/dllib_tests"
cd "${BUILD_DIR}"
if [[ -f ./tests/dllib_tests.exe ]]; then
  ./tests/dllib_tests.exe
elif [[ -x ./tests/dllib_tests || -f ./tests/dllib_tests ]]; then
  ./tests/dllib_tests
else
  echo "[dev.sh] Test executable not found: ${BUILD_DIR}/tests/dllib_tests" >&2
  exit 1
fi
