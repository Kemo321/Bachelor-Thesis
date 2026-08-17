# Chapter 1 — Introduction and setup

## 1.1 Formal abstract

The dominant practice in contemporary deep learning is to treat a framework such as PyTorch (LibTorch in C++) as an opaque runtime: the researcher expresses a computation graph, and the library decides when to allocate device memory, which kernel to launch, and when to synchronise with the host. That abstraction is scientifically productive, but it is pedagogically and experimentally incomplete. A Bachelor's thesis that claims to *understand* GPU training cannot stop at calling `loss.backward()`. It must reconstruct the contract that LibTorch hides: who owns the `cudaMalloc` buffer, which stream the GEMM is enqueued on, why a transpose of a weight matrix can cost tens of milliseconds even when the arithmetic intensity of the subsequent multiply is modest, and why a JPEG decoder on the CPU can starve an otherwise well-tuned YOLO backbone.

This project therefore constructs **DeepLearnLib**, a high-performance C++17 / CUDA deep-learning library written from scratch, with an explicit ban on `torch::Tensor` and `<torch/torch.h>` inside the core. The library is not a teaching toy that evaluates convolutions on the CPU. It is a production-shaped stack on NVIDIA's native APIs:

- **CUDA Runtime** for allocation (`cudaMalloc` / `cudaFree`), streams, events, and asynchronous copies.
- **cuBLAS** (`cublasGemmEx`) for dense matrix multiplies that implement fully-connected layers, with TensorFloat-32 allowed on Ampere and newer.
- **cuDNN** for `Conv2d`, `MaxPool2d`, `BatchNorm2d`, and the fused convolution-bias path inside `FusedCBR2d`.
- **Custom `__global__` kernels** for in-place arithmetic, fused SGD, BatchNorm-affine + LeakyReLU, and YOLOv1 loss reduction.

LibTorch remains in the repository solely as an **optional baseline**. Binaries named `*_torch` link `TorchBaseline` so that epoch time, micro-op latency, and VRAM can be compared on the same GPU, the same batch size, and the same dataset split. The scientific claim of the thesis is not that a student framework will universally outperform a decade of PyTorch engineering. The claim is narrower and stronger: once the allocation behaviour, transpose strategy, and host/device overlap are made explicit and aligned with what LibTorch already does internally, a from-scratch library can approach LibTorch wall-clock on YOLOv1 training (empirically, Custom versus Torch epoch times in the same ~76–78 ms region at batch 16 on the development GPU after the fused-SGD / GPU-loss-reduce round). That result is only interpretable if the reader can audit every `cudaMalloc` and every `to_host` on the training path.

The second architectural claim is **separation of concerns**. A generic tensor library must not contain a detector topology. YOLOv1 and the CIFAR-10 CNN live in `benchmarks/models/` and compile into `DeepLearnModels`. `DeepLearnLib` contains tensors, layers, loaders, and losses. Changing the number of YOLO classes does not relink the convolution kernels. This split is developed at length in Chapter 2.

The third claim is a **memory–time tradeoff** stated without apology. DeepLearnLib accepts a large resident VRAM footprint—on the order of **~13 GiB for a YOLOv1 training process** at the thesis batch size—because every layer caches activations, workspaces, and gradient buffers with `dl::Tensor::ensure`. After the first step of an epoch, a stable batch size performs **zero additional `cudaMalloc` on the hot path**. That is the same strategy LibTorch's caching allocator implements behind a different API. Chapter 3 treats this as the central systems decision of the project, not as an implementation detail.

The remainder of this chapter describes how the software is built: why CMake and Ninja, why ccache, which language standards are frozen, how CUDA architecture flags are pinned so that a container nvcc newer than the host driver does not emit unloadable PTX, and how Docker versus a local MSVC toolchain is supposed to be used.

## 1.2 Problem statement: replacing LibTorch on the training path

Replacing LibTorch does not mean reimplementing autograd in full generality. DeepLearnLib uses **explicit reverse-mode traversal**: each `Layer` stores what it needs during `forward`, and `backward` is invoked from the tail of the layer list to the head. There is no dynamic graph, no Python dispatcher, and no operator registry. The cost of that simplicity is that a new layer must be written as a C++ class with a documented memory contract (Chapter 4 and `ADDING_LAYERS.md`). The benefit is that the generated SASS is exactly the kernels the thesis describes.

The training path that must not go through LibTorch is:

1. Decode a mini-batch of images on the CPU (OpenCV).
2. Upload NCHW float tensors with `Tensor::from_host` onto a CUDA stream.
3. Forward through generic layers (fused conv-BN-LeakyReLU, pooling, flatten, GEMM).
4. Evaluate a GPU loss (`YOLOLoss` or `CrossEntropyLoss`).
5. Backward through the same layers.
6. Clip gradients and apply fused SGD in-place.
7. Log scalars. Logging is the *only* intended host synchronisation per batch, and even that is skipped when the epoch does not print.

Every one of those steps has a corresponding failure mode that the documentation records: a forgotten `to_host` inside `YOLOLoss` (historically `thrust::reduce`) reintroduces a device-wide sync; a physical `transpose()` before GEMM copies megabytes of weights; an unbounded thread pool for JPEG decode thrashes the CPU at CIFAR batch 64. The chapters that follow are organised around those failure modes and their remedies.

## 1.3 Language and API constraints

The workspace rule that governs the core is absolute: **no** `#include <torch/torch.h>` and **no** `torch::` in `src/` or `include/DeepLearnLib/`. Tensors allocate through `cudaMalloc` and are owned by `std::shared_ptr<float>` with `dl::CudaDeleter`. Views share that pointer. Host vectors are always IEEE-754 `float`; device storage may be `float` or `__half`.

C++17 is required (`CMAKE_CXX_STANDARD 17`, `CMAKE_CUDA_STANDARD 17`, extensions off). The library uses `std::optional` for layer caches, `std::shared_ptr` for storage, structured bindings in a few loaders, and `if constexpr` in fused kernels that exist in both FP32 and FP16 instantiations. CUDA language is enabled at the CMake project level (`project(MiniC_DL LANGUAGES CXX CUDA)`), and selected `.cpp` files are compiled *as CUDA* via `set_source_files_properties(... LANGUAGE CUDA)` so that `__global__` kernels can live next to the C++ methods that launch them without a separate `.cu` naming convention.

On GNU/Clang, host code is built with `-mavx2 -mfma` so that CPU JPEG post-processing and Highway (linked as `hwy`) can use SIMD. On MSVC, `/bigobj` and `/Zc:preprocessor` are required because the CUDA headers and the volume of template instantiations overflow the default COFF object limits; nvcc is passed `--allow-unsupported-compiler` together with the same preprocessor flag because student and CI machines often pair a newer Visual Studio with an NGC toolkit that has not yet blessed that cl.exe version.

## 1.4 Why CMake

CMake 3.18+ is the meta-build. The project is not a single `nvcc *.cu` invocation. It must:

- detect CUDAToolkit and, on CUDA 13 images, **manually import** `CUDA::cudnn` because the toolkit may ship cuDNN without an imported CMake target;
- optionally fetch or find TBB, GoogleTest, Google Benchmark, nlohmann_json, spdlog, pugixml, OpenCV;
- compile a subset of `src/*.cpp` as CUDA and the rest as CXX, then device-link with `CUDA_RESOLVE_DEVICE_SYMBOLS ON`;
- gate Torch baselines on `Torch_FOUND` / `TorchBaseline`;
- pin `CMAKE_CUDA_ARCHITECTURES` from `nvidia-smi` rather than `native` on Blackwell-class GPUs, because early toolchains map RTX 50-series cards to `sm_100` plus PTX that the installed driver refuses to JIT (`cudaErrorUnsupportedPtxVersion`).

That last point is not cosmetic. Container images (NGC PyTorch) frequently contain a **newer nvcc than the host driver**. Emitting PTX “for forward compatibility” is exactly wrong in that configuration: the driver cannot load the PTX. `scripts/cuda_env.sh` therefore sets `CMAKE_CUDA_ARCHITECTURES` to `${GPU_CC//./}-real` (for example `120-real`) and sets `TORCH_CUDA_ARCH_LIST` to the dotted compute capability so LibTorch's build does not pull in a fat `12.0+PTX` list from the NGC environment.

CMake also copies `config/` into the build tree and defines `DEEPLEARN_SOURCE_DIR` on every benchmark binary so `load_pipeline_config()` can find `experiments.json` regardless of the current working directory (binaries run from `build/benchmarks/`).

## 1.5 Why Ninja

The generator is Ninja (`cmake -G Ninja`). Ninja's DAG evaluation is faster than Unix Makefiles for a project that recompiles CUDA translation units after header edits. `scripts/dev.sh` **refuses to reuse** an existing `CMakeCache.txt` that was not produced by Ninja: it deletes the build directory. Mixing Visual Studio generators with Ninja caches is a common source of “it builds on my machine” failures on Windows.

Parallelism is implicit: `ninja -C build` uses all cores; `cmake --build build --parallel` does the same when the menu cannot find a `build.ninja`. CUDA compilation is the long pole. Ninja's ability to keep nvcc busy on independent `.cpp` files (Tensor, Conv2d, FusedCBR2d, YOLOLoss, …) is the practical reason the default CI path is Ninja rather than `cmake --build` with MSBuild.

## 1.6 Why ccache

A full Release configure of DeepLearnLib plus tests plus Torch baselines is expensive. The Docker compose file mounts a named volume `ccache_data` at `/ccache` with `CCACHE_MAXSIZE=10G`. `dev.sh` sets

```text
CMAKE_CXX_COMPILER_LAUNCHER=ccache
CMAKE_C_COMPILER_LAUNCHER=ccache
CMAKE_CUDA_COMPILER_LAUNCHER=ccache
```

when `ccache` is on `PATH`. CUDA caching is particularly valuable: nvcc's preprocessor hash is stable across “comment-only” edits of neighbouring files, and the 10 GiB cap is sized for a thesis workspace that iterates on kernels daily. `CCACHE_COMPILERCHECK=content` (Dockerfile) hashes the compiler binary so a toolkit bump inside the NGC image does not silently reuse objects compiled by a different nvcc.

ccache does **not** cache linking. Device linking (`CUDA_RESOLVE_DEVICE_SYMBOLS`) still runs. The win is compile time, not the final `libDeepLearnLib` link.

## 1.7 CUDA architecture pinning in CMake

Root `CMakeLists.txt` queries

```text
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

and, outside CI, writes `CMAKE_CUDA_ARCHITECTURES` to `<cc><minor>-real`. CI (GitHub Actions) uses a portable list `70;75;80;86;89;90` and forces `TORCH_CUDA_ARCH_LIST=7.5` so hosted runners without a GPU still compile SASS for a Turing-class virtual architecture. If the user passes `-DCMAKE_CUDA_ARCHITECTURES=native` on CMake ≥ 3.24, the file **overrides** that to the detected `-real` arch when `nvidia-smi` succeeded, for the PTX reason above.

`scripts/menu.sh` additionally compares the cached arch against the live GPU and re-runs `cmake` if they diverge—for example after moving the same bind-mount from an Ada laptop to a Blackwell workstation.

## 1.8 Docker environment

`Dockerfile` is based on `nvcr.io/nvidia/pytorch:26.03-py3`. That choice is deliberate and slightly ironic: the image contains LibTorch, which the *core* must not use, but it also contains a coherent CUDA, cuDNN, and Python stack. The Dockerfile then installs CMake, Ninja, ccache, pugixml, OpenCV, TBB, clang-tidy, cppcheck, Doxygen, and Graphviz. Pillow / opencv-python-headless are installed only if the base image's Python cannot import them.

`docker-compose.yml` defines service `yolo-app`:

- image tag `yolo-bachelor-thesis:latest`, container name `yolo_dev_container`;
- `shm_size: 32gb` because DataLoader workers and OpenCV decode can exhaust the default 64 MiB `/dev/shm`;
- one NVIDIA GPU via Compose `deploy.resources.reservations.devices`;
- environment `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`;
- bind-mount of the repository at `/app`;
- named volume `build_cache` at `/app/build` so configure products survive container recreation;
- named volume `ccache_data` at `/ccache`;
- `working_dir: /app`, TTY, `command: ["/bin/bash"]`.

Canonical workflow:

```bash
docker compose up -d --build
docker exec -it yolo_dev_container bash
./scripts/dev.sh
./scripts/menu.sh
```

Inside the container, `dev.sh` mirrors GitHub Actions: Ninja, `USE_CUDA=ON`, ccache launchers, then `dllib_tests`. The menu builds individual benchmark targets on demand.

## 1.9 Local MSVC / Ninja setup (Windows)

The authoring machine is Windows. WSL `bash.exe` from `System32` is **not** a supported compiler environment: it does not see MSVC's `INCLUDE`. `scripts/dev.sh` detects MSYS/Cygwin and exits if `INCLUDE` is unset, directing the user to:

```powershell
powershell -File scripts/dev.ps1
```

`dev.ps1` is responsible for loading `vcvars64` and then invoking Git Bash so that `dev.sh` sees a real `cl.exe` and CUDA host compiler. CMake on Windows still uses `-G Ninja`. MSVC-specific flags (`/bigobj`, `/Zc:preprocessor`, nvcc `--allow-unsupported-compiler`) are applied in `src/CMakeLists.txt`. cuDNN DLLs found under `CUDNN_ROOT/bin` are copied next to `dllib_tests` on Windows so the loader does not pick an older cuDNN from a Torch install on `PATH`.

Do not mix a Visual Studio generator cache with Ninja. If `CMakeCache.txt` exists and does not mention Ninja, `dev.sh` deletes `build/`.

## 1.10 Sanitizers, numerics, and CI

Debug builds on non-MSVC enable ASan and UBSan on `DeepLearnLib` (`USE_SANITIZERS`, default ON for Debug). These are host sanitizers; they will not catch CUDA races. `DEBUG_NUMERICS` (CMake option) injects NaN/Inf checks after layer passes and is off by default because it synchronises.

GitHub Actions configures Release with Ninja, CUDA arch 75, and builds `dllib_tests` plus a set of **Custom** benchmark binaries (`bench_voc_custom`, `short_voc_custom`, `overfit_voc_custom`, `train_synthetic_custom`, `train_tabular_custom`, `train_cifar_custom`). Runners have no GPU; CI only proves linkage. GPU numbers in the thesis always come from the local container.

## 1.11 What “from scratch” does and does not mean

From scratch means: the *control flow* of training, the *ownership* of device buffers, the *choice* of cuDNN descriptors, and the *text* of elementwise kernels are authored in this repository. It does not mean reimplementing SGEMM or Winograd convolution. Using cuBLAS and cuDNN is the correct engineering decision; LibTorch does the same. The thesis contribution is the systems integration—allocation policy, fusion, overlap—and the empirical comparison against LibTorch on identical workloads, not a faster GEMM than NVIDIA.

## 1.12 C++17 and CUDA language flags in detail

The root `CMakeLists.txt` freezes both host and device language at ISO C++17:

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

`CXX_EXTENSIONS OFF` is not a style preference. GNU extensions (`gnu++17`) change the preprocessor, enable non-portable `__int128` paths in some headers, and can silently diverge from MSVC. The thesis binaries must be comparable across Docker (GCC inside NGC) and Windows (MSVC + nvcc). ISO C++17 is the intersection that both compilers actually honour.

C++17 is the minimum that makes the tensor and loader APIs honest:

- `std::optional<Tensor>` is the cache slot type. A raw pointer would not express “unallocated until first `ensure`.” A `unique_ptr` would forbid returning `as_view()` aliases. Optional-plus-move is the ownership model Chapter 3 depends on.
- `std::shared_ptr<float>` with a custom deleter is how `CudaDeleter` and `PinnedHostDeleter` participate in RAII without writing a homegrown intrusive refcount.
- Structured bindings appear in loaders when unpacking `(images, targets)`.
- `if constexpr` selects FP32 versus `__half` kernel bodies in `FusedCBR2d.cpp` and `Tensor.cpp` without preprocessor `#ifdef` soup inside `__global__` templates.
- `std::launch::async` and `std::future` are the prefetch contract in Chapter 5. C++11 already had them; C++17’s guaranteed copy elision and `std::optional` make the surrounding API tolerable.

CUDA language is set to the same 17. nvcc’s host-side parser must accept the same headers that `cl.exe` or `g++` parse. Mixing `CMAKE_CXX_STANDARD 17` with an implicit CUDA 14 host dialect is a historically common source of “it compiles as CXX and fails as CUDA” errors on `std::optional` in headers included from `.cpp` files that `set_source_files_properties(... LANGUAGE CUDA)` reinterprets.

`src/CMakeLists.txt` repeats the standards on the target itself:

```cmake
set_target_properties(DeepLearnLib PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_STANDARD 17
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
```

`CUDA_RESOLVE_DEVICE_SYMBOLS ON` is mandatory because `__global__` kernels live in multiple translation units (`Tensor.cpp`, `YOLOLoss.cpp`, `FusedCBR2d.cpp`, …). Without device-link, the host objects contain unresolved device symbols and the final executable fails at load or at first launch with an unhelpful “invalid device function.” Device linking is the CUDA analogue of a static archive of `.o` files: it is slow, it is not cached by ccache, and it is the reason incremental *compile* wins from Ninja still leave a noticeable *link* tail.

### 1.12.1 GNU/Clang host flags

On GCC and Clang, DeepLearnLib is compiled with `-mavx2 -mfma`. Those flags do not affect device SASS. They exist because:

1. OpenCV’s JPEG decode and colour conversion on the CPU can use SIMD when the compiler is allowed to emit FMA.
2. The library links Highway (`hwy`) as a public dependency; Highway’s dispatch expects a host that at least *can* execute AVX2 on the development machines used for the thesis.
3. `_GLIBCXX_USE_CXX11_ABI=1` is forced so that a LibTorch wheel built against the new libstdc++ ABI does not produce `std::string` dual-ABI link errors when Torch baselines are enabled. Custom core does not need Torch, but the *same* CMake tree builds both, so the ABI must be one.

### 1.12.2 MSVC / nvcc host flags

On MSVC the core is compiled with:

```text
CXX:   /bigobj /Zc:preprocessor
CUDA:  --allow-unsupported-compiler -Xcompiler=/Zc:preprocessor,-bigobj
```

`/bigobj` raises the COFF section limit. CUDA headers, spdlog, and the volume of explicitly instantiated FP32/FP16 kernels overflow the historical 65 536-section default. Without `/bigobj`, the failure is a cryptic `C1128` at the end of an otherwise successful nvcc run.

`/Zc:preprocessor` enables a standard-conforming preprocessor. CUDA’s `__host__` / `__device__` macros and Microsoft’s traditional preprocessor disagree about token pasting. nvcc is told the same flag via `-Xcompiler` so the *device* compilation’s host pass sees the same preprocessor the *host* compilation used.

`--allow-unsupported-compiler` exists because student machines and GitHub-hosted images frequently pair a Visual Studio newer than the NGC toolkit’s blessed `cl.exe` table. The alternative is “the project does not configure.” The flag is a documented risk: NVIDIA does not test that pairing. The mitigation is Docker for the numbers that go into the thesis, and MSVC locally only for iteration.

OpenCV 4.10 Windows packs advertise `vc16` / `vc17` runtimes. Visual Studio 2026 reports `MSVC_VERSION >= 1950`, which OpenCVConfig.cmake historically skipped. Root CMake therefore forces `OpenCV_RUNTIME vc16` on those toolchains so `find_package(OpenCV)` succeeds. This is a packaging workaround, not a claim that OpenCV was rebuilt for that compiler.

### 1.12.3 Why the sources are `.cpp` compiled as CUDA

A conventional layout would put kernels in `foo.cu` and host logic in `foo.cpp`. DeepLearnLib instead lists Tensor, Conv2d, FusedCBR2d, YOLOLoss, and related files in `DEEPLEARN_CUDA_SOURCES` and sets `LANGUAGE CUDA` on them. The reason is cohesion: the `__global__` kernel that implements `sgd_update_` sits in the same file as `Tensor::sgd_update_`, so a reader (and a thesis examiner) can audit launch configuration and C++ wrapper in one place. The cost is that every edit of those files pays an nvcc compile. That cost is why YOLO and SimpleCNN were removed from this list (Chapter 2).

Files that remain CXX-only include `Flatten.cpp` (a view), `Logger.cpp`, `CSVLoader.cpp`, `mAP.cpp`, and the OpenCV loaders’ non-kernel portions where they are not in the CUDA list. `dataset.cpp` and `ClassificationLoader.cpp` are host-only; they launch no kernels.

## 1.13 CMake as a scientific instrument, not a convenience

CMake 3.18 is the floor because CUDA as a first-class `enable_language(CUDA)` language, imported `CUDA::cudart` / `CUDA::cublas` targets, and `CUDA_ARCHITECTURES` exist in a usable form from that version. The project is named `MiniC_DL` at the CMake level for historical reasons; the library target is `DeepLearnLib`.

Configure must do work that a handwritten Makefile would get wrong:

1. **Toolkit discovery.** `CUDAToolkit_ROOT` is taken from `CUDA_PATH` on Windows or `/usr/local/cuda` on Linux. CUDA 13 NGC images may ship cuDNN without an imported `CUDA::cudnn` target. CMake then `find_path`/`find_library`s `cudnn.h` and `cudnn` / `cudnn64_9` / `cudnn64_8`, and on Windows globs `cudnn_*.lib` so the split cuDNN 8/9 libraries (`cudnn_ops`, `cudnn_cnn`, …) are linked as INTERFACE dependencies of `CUDA::cudnn`. A missing `CUDNN_ROOT` is a `FATAL_ERROR`, not a silent CPU fallback. There is no CPU training path.

2. **Optional LibTorch.** `find_package(Torch QUIET)` is attempted only if MSVC has `INCLUDE` set (otherwise Caffe2’s CUDA probe dies on `stdlib.h` and *kills the entire configure*, including Custom tests). After Torch is found, LibTorch’s CMake **overwrites** `CMAKE_CUDA_ARCHITECTURES`. Root CMake therefore snapshots `DEEPLEARN_CUDA_ARCHITECTURES` before `find_package(Torch)` and **forces it back**. Without that restore, a carefully pinned `120-real` becomes Torch’s fat `7.5;8.0;8.6;9.0;12.0+PTX` list and the PTX problem in §1.7 returns.

3. **OpenCV as a gate.** If OpenCV is missing, dataset loaders and `benchmarks/` are skipped. Unit tests that only need `dl::Tensor` still build. This is how CI can compile the core without a full vision stack, and how a developer without VOC still iterates on GEMM.

4. **Config files in the build tree.** `file(COPY config/ …)` plus `DEEPLEARN_SOURCE_DIR` compile definitions on every benchmark binary make `experiments.json` locatable whether the process cwd is `build/benchmarks/` or the repo root. Thesis plots must not depend on “remember to `cd` first.”

5. **Kineto disabled.** `CAFFE2_USE_KINETO OFF` prevents Torch’s profiler library from entering the link line of Custom binaries through a transitive Caffe2 dependency.

## 1.14 Ninja versus other generators, at the level of the DAG

Ninja is not chosen because it is fashionable. It is chosen because CUDA compilation is a high-latency, highly parallel, header-sensitive workload.

A typical Release configure compiles on the order of a dozen nvcc translation units for the core, plus YOLO/SimpleCNN as CXX, plus every `train_*` / `inference_*` binary, plus GoogleTest, plus (optionally) Torch baselines. Unix Makefiles evaluate the DAG with recursive make; the latency of `stat` and the inability to start the next nvcc until a recipe shell exits dominate. Ninja keeps a persistent process, understands restat, and will start `Tensor.cpp` and `YOLOLoss.cpp` on two cores in the same millisecond.

`scripts/dev.sh` **deletes** `build/` if `CMakeCache.txt` exists and does not contain `Ninja`. Mixing generators is not a theoretical concern on this repository: Windows developers naturally open the folder in Visual Studio, which writes a VS cache; the next `dev.sh` would then invoke MSBuild against flags that were written for Ninja’s `build.ninja`, or vice versa. The failure modes are “no rule to make target” and silently stale CUDA objects. The script’s policy is: one generator, always Ninja, cache mismatch is fatal to the cache not to the user.

Parallelism: `ninja -C build` uses all logical cores by default. CUDA compiles are memory-heavy; a 16-core machine compiling twelve nvcc jobs can exceed RAM if each nvcc also spawns host `cl`/`g++`. In practice the thesis machines have been stable. If they were not, `ninja -jN` would be the knob, not a switch back to Makefiles.

CI uses the same generator (`cmake -G Ninja`) so that “green on GitHub, red locally” cannot be blamed on Make versus Ninja recipe differences.

## 1.15 ccache: what hashes, what does not, why 10 GiB

The Docker image and compose file both set:

```text
CCACHE_DIR=/ccache
CCACHE_MAXSIZE=10G
CCACHE_COMPRESS=1
CCACHE_COMPILERCHECK=content
CMAKE_{C,CXX,CUDA}_COMPILER_LAUNCHER=ccache
```

`CMAKE_*_COMPILER_LAUNCHER` is the CMake-supported way to wrap the compiler. It is superior to aliasing `nvcc` in `PATH` because CMake still knows the real compiler for feature tests.

**What ccache hashes for CUDA.** The preprocessor output of the translation unit, the compiler flags, and (with `COMPILERCHECK=content`) the *bytes* of the compiler binary. A comment-only change in a `.cpp` that does not affect the preprocessed output is a cache hit. An NGC image bump that replaces `/usr/local/cuda/bin/nvcc` is a cache miss for every CUDA file, which is the correct behaviour: SASS from nvcc 12.8 must not be reused under nvcc 13.0.

**What ccache does not hash.** The device-link step, the host link of `libDeepLearnLib`, and the final executable. After a cache-hot compile, the user still waits on `nvlink` / `dlink` and `lld`/`link.exe`. That is why “ccache 100% hit” is not “instant binary.”

**Why 10 GiB.** A single CUDA object for `Tensor.cpp` in Release with debug info off is already large; Debug objects are larger; FP32 and FP16 instantiations live in the same TU. A week of kernel iteration across Debug and Release, plus Torch baseline objects, plus a toolkit bump that duplicates the cache, fills several gigabytes. `CCACHE_COMPRESS=1` trades CPU for disk. The named Docker volume `ccache_data` survives `docker compose down` so that recreating the container does not throw away a day’s compiles. Bind-mounting the repo at `/app` does **not** persist `/app/build`; that is a second named volume `build_cache`, because putting the build tree in the bind mount would let a Windows host see Linux ELF objects and vice versa.

GitHub Actions uses `hendrikmuhs/ccache-action` with the same three launcher variables. CI’s win is across workflow runs on the same runner cache, not across developers.

## 1.16 SASS versus PTX: the Blackwell / NGC failure mode

NVIDIA’s compilation model has two device artefacts:

- **SASS** (`sm_XX`, CMake suffix `-real`): machine code for a specific compute capability. The driver loads it directly.
- **PTX** (`compute_XX`, CMake suffix `-virtual`, or a `+PTX` entry in `TORCH_CUDA_ARCH_LIST`): an intermediate ISA. The driver **JITs** PTX to SASS at process start. JIT requires a driver that understands that PTX version.

NGC PyTorch images ship a *new* nvcc. A host laptop or workstation often has a *slightly older* driver. nvcc, asked for `native` or for a fat bin with `12.0+PTX`, emits PTX that the host driver rejects with `cudaErrorUnsupportedPtxVersion`. Early Blackwell (RTX 50-series) toolchains compounded this by mapping the card to `sm_100` plus PTX rather than the actual `sm_120`.

The project’s policy, implemented in three cooperating places, is **never emit PTX for the local GPU**:

1. `scripts/cuda_env.sh` reads `nvidia-smi --query-gpu=compute_cap`, sets `CMAKE_CUDA_ARCHITECTURES=${GPU_CC//./}-real` (e.g. `12.0` → `120-real`), and overwrites NGC’s `TORCH_CUDA_ARCH_LIST` with the dotted capability only (`12.0`), unless `KEEP_TORCH_CUDA_ARCH_LIST` is set.
2. Root `CMakeLists.txt` runs the same `nvidia-smi` query at configure time, writes `<cc>-real` into the cache, and **overrides** a user-supplied `native` when detection succeeded.
3. `scripts/menu.sh` compares the cached arch to the live GPU and re-invokes CMake if the machine changed (Ada laptop vs Blackwell workstation, same bind-mount).

CI cannot see a GPU. It therefore compiles a portable real-SASS list `70;75;80;86;89;90` and sets `TORCH_CUDA_ARCH_LIST=7.5` so hosted runners still produce Turing-class objects. Those objects will not run on the runner; they exist to prove the tree *links*.

The thesis numbers always come from a container whose nvcc and whose *driver* (the host’s) have been verified to load the `-real` cubin. That verification is: `./scripts/dev.sh` then `./tests/dllib_tests` on GPU.

## 1.17 Docker environment, layer by layer

### 1.17.1 Base image

`Dockerfile` starts from `nvcr.io/nvidia/pytorch:26.03-py3`. The irony is acknowledged in the abstract: the image contains LibTorch, which `src/` must never include. The engineering reason is coherence. NVIDIA’s PyTorch containers pin:

- a CUDA toolkit,
- a cuDNN matching that toolkit,
- a Python with `torch` importable for *baseline* binaries,
- NCCL, HPC-X, and driver stubs that the NVIDIA Container Toolkit maps onto the host GPU.

Building a “pure CUDA” image from `nvidia/cuda` and then installing matching cuDNN, OpenCV, and a LibTorch wheel by hand is how version skew enters a thesis. One NGC tag is one known-good tuple. Custom code simply does not `#include <torch/torch.h>`.

### 1.17.2 Packages installed on top of NGC

`apt-get` adds CMake, Ninja, ccache, git, pkg-config, pugixml, OpenCV, TBB, clang-tidy, cppcheck, Doxygen, Graphviz. Pillow and `opencv-python-headless` are installed only if the base Python cannot import them—NGC sometimes already has them, and a second pip install would fight the image.

`CCACHE_COMPILERCHECK=content` is set in `ENV` so that even a developer who forgets compose still hashes the compiler.

### 1.17.3 Compose service `yolo-app`

| Compose key | Rationale |
| --- | --- |
| `image: yolo-bachelor-thesis:latest` | Stable tag for `docker exec` scripts and documentation. |
| `container_name: yolo_dev_container` | `docker exec -it yolo_dev_container bash` does not require looking up a hash. |
| `shm_size: 32gb` | DataLoader workers, OpenCV, and `std::async` JPEG threads use POSIX shared memory. Docker’s default 64 MiB `/dev/shm` causes mysterious `bus error` or allocator failures under batch 16 VOC. 32 GiB is generous relative to the host; it is a *limit*, not a reservation of 32 GiB RAM. |
| `deploy.resources.reservations.devices` NVIDIA `count: 1` | Compose GPU reservation; requires NVIDIA Container Toolkit. |
| `NVIDIA_VISIBLE_DEVICES=all` | The process sees the GPU. |
| `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` | `compute` for CUDA, `utility` for `nvidia-smi` inside the container (arch pinning), `graphics` for some OpenCV/GL code paths. |
| bind-mount `.` → `/app` | Edit on the Windows or Linux host, compile in Linux. |
| volume `build_cache` → `/app/build` | Linux objects must not be written into the Windows-visible tree as the only copy; the named volume keeps ELF objects across container recreation. |
| volume `ccache_data` → `/ccache` | As above. |
| `command: ["/bin/bash"]` + TTY | Interactive development container, not a one-shot train job. |

Canonical inner-container workflow remains:

```bash
docker compose up -d --build
docker exec -it yolo_dev_container bash
./scripts/dev.sh
./scripts/menu.sh
```

`dev.sh` is deliberately aligned with GitHub Actions: Ninja, `USE_CUDA=ON`, ccache launchers, build `dllib_tests`. The menu then builds individual benchmark targets so a VOC train does not require compiling every Torch binary first.

## 1.18 Local MSVC / Ninja setup on Windows

The authoring machine for this repository is Windows. Two bash implementations exist; only one is valid.

**Invalid:** `C:\Windows\System32\bash.exe` (WSL). It does not inherit MSVC’s `INCLUDE` / `LIB`. nvcc’s host pass then fails to find `stdlib.h`. Caffe2’s Torch probe, if reached, `FATAL_ERROR`s the entire CMake configure.

**Valid:** Git Bash (`C:\Program Files\Git\bin\bash.exe`) *after* `vcvars64.bat` has exported the Visual C++ environment.

`scripts/dev.ps1` automates that pairing:

1. Locate Visual Studio with `vswhere.exe` (`-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64`).
2. Fall back to a hard-coded VS 18 Community path if vswhere is missing.
3. `cmd.exe /c "call vcvars64.bat && git-bash scripts/dev.sh"`.

`dev.sh` still defends itself: if `MSYSTEM` / MSYS / Cygwin is detected and `INCLUDE` is empty, it prints the PowerShell one-liner and exits 1.

CMake on Windows is still `-G Ninja`, not `-G "Visual Studio 17 2022"`. Ninja invokes `cl.exe` and `nvcc.exe` as the CMake CUDA toolchain expects. The Visual Studio *IDE* can be used as an editor; it must not be allowed to overwrite `CMakeCache.txt` with a VS generator. If it does, `dev.sh` wipes `build/`.

cuDNN DLLs discovered under `CUDNN_ROOT/bin` are copied next to `dllib_tests` on Windows. The Windows loader searches the executable directory before `PATH`. A leftover LibTorch `cudnn64_8.dll` on `PATH` would otherwise load an ABI-incompatible library into a Custom process.

## 1.19 Sanitizers, numerics, and what CI actually proves

Debug builds on non-MSVC may enable ASan and UBSan on `DeepLearnLib` (`USE_SANITIZERS`, default ON for Debug in `dev.sh`). These instrument **host** code: vector out-of-bounds in a loader, use-after-free of a `std::future`, signed overflow in index arithmetic. They do **not** instrument CUDA kernels. A race between two streams, or a buffer overrun in `fused_bn_leaky_kernel`, will not be reported. Device correctness is GoogleTest on GPU plus the overfit pipeline.

`DEBUG_NUMERICS` (CMake option, default OFF) injects NaN/Inf scans after layer forward/backward. Those scans synchronise the device. They are a debugging aid, not a performance configuration. Thesis timings use Release without this flag.

GitHub Actions (`.github/workflows/ci.yml`) configures Release, Ninja, `CMAKE_CUDA_ARCHITECTURES=75`, ccache launchers, and builds `dllib_tests` plus Custom binaries (`bench_voc_custom`, `short_voc_custom`, `overfit_voc_custom`, `train_synthetic_custom`, `train_tabular_custom`, `train_cifar_custom`). Runners have no GPU. CI proves:

- the tree configures on Ubuntu with CUDA toolkit packages,
- nvcc accepts the sources as CUDA 17,
- Custom targets link without Torch,
- a compile-only regression did not land.

CI does **not** prove epoch times, VRAM, or numerical agreement with LibTorch. Those measurements are local, containerised, and recorded in `results/*/metrics_*.csv` (Chapter 6).

## 1.20 What “from scratch” does and does not mean

From scratch means: the *control flow* of training, the *ownership* of device buffers, the *choice* of cuDNN descriptors, the *text* of elementwise kernels, the *prefetch* of JPEG batches, and the *comparison harness* against LibTorch are authored in this repository.

It does not mean reimplementing SGEMM, Winograd convolution, JPEG, or XML. Using cuBLAS, cuDNN, OpenCV, and pugixml is the correct engineering decision; LibTorch uses the first two as well. The thesis contribution is systems integration—allocation policy, fusion, host/device overlap—and the empirical comparison on identical workloads, not a faster GEMM than NVIDIA.

A reader who wants a layer tutorial rather than architecture should read `docs/ADDING_LAYERS.md` after Chapter 3.

## 1.21 Reading guide for the remaining chapters

- **Chapter 2** explains why YOLO is not in `libDeepLearnLib`, how `DeepLearnModels` is linked, and why that split exists for compile time as much as for purity.
- **Chapter 3** is the memory model every later optimisation assumes: `CudaDeleter`, `ensure`, the ~13 GiB VRAM tradeoff, and `matmul_into` with logical `CUBLAS_OP_T`.
- **Chapter 4** is the compute model: custom `__global__` kernels, `FusedCBR2d`, and the single-pass `YOLOLoss` reduction that replaced `thrust::reduce`.
- **Chapter 5** is the I/O model: bounded JPEG threads, `std::future` prefetch, and double-buffered CUDA streams.
- **Chapter 6** is how those claims are measured: Google Benchmark, CUDA Events, CSV columns, and `plot_metrics.py`.

Read them in order. Later chapters assume the vocabulary of earlier ones and cite source files by name so that an examiner can grep the tree.
