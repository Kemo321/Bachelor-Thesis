# Build

This page is how to compile `DeepLearnLib` and `dllib_tests`. Running training binaries, the menu, and Docker-as-a-workbench are in [Usage — running](../usage/running.md).

## Requirements

- NVIDIA GPU and a driver that can load the cubin you compile
- CMake 3.18 or newer, Ninja, a C++17 compiler
- CUDA toolkit with nvcc, cuBLAS, and cuDNN
- OpenCV and pugixml for the loaders in `dataset.cpp`, `utils.cpp`, and `ClassificationLoader.cpp`

Without OpenCV, CMake still builds the core sources that do not decode images. `benchmarks/` is skipped. Unit tests that only need `dl::Tensor` still build.

LibTorch is optional and is not linked into `DeepLearnLib`.

```bash
cmake -S . -B build -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/tests/dllib_tests
```

The CMake project name is `MiniC_DL`. The library target is `DeepLearnLib`. On Windows, load the Visual C++ environment before configuring, and keep the Ninja generator. `scripts/dev.ps1` does that load and then runs `scripts/dev.sh`. A Visual Studio generator cache is discarded by `dev.sh`, because mixing generators leaves stale CUDA objects.

## Device architecture

Configure queries `nvidia-smi` and sets `CMAKE_CUDA_ARCHITECTURES` to `<major><minor>-real` (SASS for that GPU, no PTX). PTX would be JIT-compiled by the driver. A toolkit newer than the host driver emits PTX the driver rejects (`cudaErrorUnsupportedPtxVersion`).

CI has no GPU. It passes an explicit SASS list so the sources still compile. Those cubins are a link check. `scripts/verify_gpu.sh` and `scripts/verify_gpu.ps1` run `dllib_tests` only on a machine where `nvidia-smi` succeeds, and they refuse to run when `GITHUB_ACTIONS` is set. They are not a CI job.

## Language

Host and device standards are C++17 with extensions off. Sources that contain kernels are `.cpp` files compiled as CUDA (`DEEPLEARN_CUDA_SOURCES`). `CUDA_RESOLVE_DEVICE_SYMBOLS` is on, because kernels live in more than one translation unit.

GCC and Clang host code uses AVX2 and FMA. That affects CPU decode, not device SASS. MSVC adds `/bigobj` and `/Zc:preprocessor`. nvcc is passed `--allow-unsupported-compiler` when the host compiler is newer than the toolkit's supported list. On Windows, cuDNN DLLs discovered under `CUDNN_ROOT` are copied next to `dllib_tests` so the loader does not bind a different cuDNN from `PATH`.

## Options that change the binary

| Option | Effect |
| --- | --- |
| `CMAKE_BUILD_TYPE=Release` | the configuration used for timings |
| `DEBUG_NUMERICS=ON` | NaN and Inf checks after layer passes; synchronises the device |
| `USE_SANITIZERS` | ASan and UBSan on host code for Debug, non-MSVC. Kernels are not instrumented |
| `USE_CUDA=ON` | required |

`dllib_tests` checks tensor math, layer shapes, and small numeric gradients. A change to `YOLOLoss` or a loader still needs an application-level run; that is outside this page.
