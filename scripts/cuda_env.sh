# Shared GPU arch pinning for Docker / local Linux.
# Container nvcc is often newer than the host driver, so we emit SASS only
# (`NNN-real`) and never PTX that the driver cannot JIT.
#
# Source this file; it may set CMAKE_CUDA_ARCHITECTURES and TORCH_CUDA_ARCH_LIST.
# shellcheck shell=bash

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
  if [[ "${GPU_CC}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    if [[ -z "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
      CMAKE_CUDA_ARCHITECTURES="${GPU_CC//./}-real"
    fi
    # NGC images export a fat TORCH_CUDA_ARCH_LIST (… 12.0+PTX).
    if [[ -z "${KEEP_TORCH_CUDA_ARCH_LIST:-}" ]]; then
      TORCH_CUDA_ARCH_LIST="${GPU_CC}"
    fi
  fi
fi
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
