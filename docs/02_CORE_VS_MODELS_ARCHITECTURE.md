# Chapter 2 — Core versus models: separation of concerns

## 2.1 The principle

A deep-learning *framework* is a reusable engine. A deep-learning *application* is a particular function from tensors to tensors with a particular loss. Conflating the two produces a library that must be relinked when the detector's class count changes, and a binary that carries unused convolution algorithm pickers into a tabular MLP demo.

DeepLearnLib enforces a strict split:

```text
┌──────────────────────────────────────────────────────────────┐
│  Applications                                                │
│  benchmarks/train_*  inference_*  overfit_*  short_*         │
│  tests/test_yolo.cpp  tests/test_simplecnn.cpp               │
│           │                                                  │
│           │  compose Layer objects                           │
│           ▼                                                  │
│  DeepLearnModels  (STATIC)                                   │
│  benchmarks/models/YOLO.cpp                                  │
│  benchmarks/models/SimpleCNN.cpp                             │
└──────────────────────────┬───────────────────────────────────┘
                           │ link
┌──────────────────────────▼───────────────────────────────────┐
│  DeepLearnLib  (STATIC / objects in src/)                    │
│  Tensor  Layer  Conv2d  FusedCBR2d  FullyConnected           │
│  MaxPool2d  BatchNorm2d  Dropout  Softmax  Flatten           │
│  YOLOLoss  CrossEntropyLoss  Network  loaders  Profiler      │
│  CUDA / cuBLAS / cuDNN / custom kernels                      │
└──────────────────────────────────────────────────────────────┘
```

The arrows are one-way. `src/YOLO.cpp` does not exist. `include/DeepLearnLib/YOLO.hpp` does not exist. A translation unit in the core may not `#include "YOLO.hpp"`. The reverse is required: `benchmarks/models/YOLO.cpp` includes `FusedCBR2d.hpp`, `FullyConnected.hpp`, `Dropout.hpp`, and so on.

## 2.2 What belongs in the core

The core answers questions of the form “how is this *kind* of computation performed on a GPU?”:

| Kind | Core type | Backend |
| --- | --- | --- |
| Dense storage | `dl::Tensor` | `cudaMalloc`, views, GEMM, in-place kernels |
| Learnable map | `Layer` subclasses | cuDNN, cuBLAS, fused CUDA |
| Detection loss (grid) | `YOLOLoss` | custom kernels, GPU reduce |
| Classification loss | `CrossEntropyLoss` | GPU |
| Ordered training helper | `Network` | explicit backward loop, binary I/O |
| Image batching | `CustomDataLoader`, `ClassificationLoader` | OpenCV + thread pool + H2D |
| Timing | `Profiler` | `cudaEvent`, `cudaMemGetInfo` |

`YOLOLoss` remains in the core even though its *name* mentions YOLO. It is a **loss function on a 7×7×(10+C) grid**, analogous to how `CrossEntropyLoss` is a loss on logits. It does not construct a backbone. `Network::fit()` uses `YOLOLoss` as a convenience for detection-shaped tensors; classification binaries never call `fit()` and instead run their own loop with `CrossEntropyLoss`. That is an acknowledged impurity (a fully generic `Network` would take a `Loss` strategy object). It is still not a topology: no layer list is hard-coded in `Network.cpp`.

## 2.3 What is forbidden in the core

A **topology** is a particular wiring of layers: “7×7 convolution, stride 2, then pool, then four residual-style 1×1/3×3 pairs, then a 4096-unit head.” That wiring is YOLO. “Two 3×3 convolutions with 16 then 32 channels, two 2×2 pools, one linear classifier” is SimpleCNN. Both are **applications**. Encoding them in `DeepLearnLib` would mean:

1. **Binary size.** Every consumer of the library—tabular MLP, unit tests that only exercise `Tensor::add_`, a future segmentation head—would link FusedCBR blocks they never call. The linker can drop unused *functions* with `--gc-sections` in some configurations, but it cannot drop the **cuDNN algorithm heuristics and static workspaces** that constructors of unused types pull in if those types are referenced from a unity translation unit. More importantly, CMake would still *compile* `YOLO.cpp` on every core rebuild.

2. **Recompilation blast radius.** Editing the YOLO head's dropout probability currently recompiles `benchmarks/models/YOLO.cpp` and relinks `DeepLearnModels` and the YOLO binaries. It does **not** recompile `Conv2d.cpp`, `Tensor.cpp`, or `YOLOLoss.cpp`. When those files lived in `src/`, they were members of `DEEPLEARN_SOURCES`; a one-line topology change invalidated the `DeepLearnLib` object graph and forced every test and every benchmark to relink the core.

3. **Layering violation.** A generic `FullyConnected` must not know that YOLO flattens `7*7*1024`. That constant belongs in the model file that chose a 7×7 grid.

4. **Thesis narrative.** The claim “we built a framework” is false if the framework *is* YOLOv1 with a tensor class attached. The claim is true if YOLOv1 is one client among CIFAR, tabular, BCCD, and synthetic.

## 2.4 How CMake realises the split

After `add_subdirectory(src)` the root `CMakeLists.txt` defines:

```cmake
add_library(DeepLearnModels STATIC
  benchmarks/models/YOLO.cpp
  benchmarks/models/SimpleCNN.cpp)
target_include_directories(DeepLearnModels PUBLIC
  "${CMAKE_SOURCE_DIR}/benchmarks/models")
target_link_libraries(DeepLearnModels PUBLIC DeepLearnLib)
```

`src/CMakeLists.txt` lists Tensor, layers, `YOLOLoss`, loaders, Profiler—**not** `YOLO.cpp` or `SimpleCNN.cpp`. Those filenames are absent from `DEEPLEARN_SOURCES` and `DEEPLEARN_CUDA_SOURCES`.

Benchmark executables link **both**:

```text
DeepLearnLib  +  DeepLearnModels  +  OpenCV  +  pugixml  +  nlohmann_json
```

Torch executables additionally link `TorchBaseline` and `${TORCH_LIBRARIES}`. `dllib_tests` links `DeepLearnModels` because `test_yolo.cpp` and `test_simplecnn.cpp` instantiate the application types. Tests that only need `dl::Tensor` still link `DeepLearnModels`; the extra `.o` files are small compared to cuDNN. The important property is that **changing SimpleCNN does not rebuild Tensor.obj**.

Public include path for models is `benchmarks/models/`, so application code writes:

```cpp
#include "YOLO.hpp"
#include "SimpleCNN.hpp"
```

and never:

```cpp
#include "DeepLearnLib/YOLO.hpp"   // does not exist
```

`dllib_configure_bench_target` also adds `benchmarks/models` to the private include path of every benchmark binary so the same headers resolve even if a target forgot to link `DeepLearnModels`—but forgetting the link would then fail at the YOLO constructor, which is the correct failure.

## 2.5 Anatomy of the YOLO application model

`benchmarks/models/YOLO.cpp` is a **constructor that pushes `std::shared_ptr<Layer>`**. It does not implement convolution. The backbone is a sequence of `FusedCBR2d` blocks (generic core) and `MaxPool2d` (generic core). The head is `Flatten`, `FullyConnected(7*7*1024, 4096)`, `LeakyReLU`, `Dropout(0.5)`, `FullyConnected(4096, 7*7*(10+num_classes))`.

`YOLO::forward` is a trivial loop:

```cpp
dl::Tensor current = input_tensor.view(input_tensor.get_shape());
for (auto& layer : backbone_layers)
    current = layer->forward(current, stream).view(...);
for (auto& layer : head_layers)
    current = layer->forward(current, stream).view(...);
```

`view` after each layer re-wraps the cache pointer without a D2D copy when shapes match. `get_all_layers()` concatenates backbone and head so `Network` can run a uniform backward.

The class is not a `torch::nn::Module`. There is no parameter dictionary beyond what each `Layer::get_parameters()` already exposes. Serialisation is `Network::save` on that flattened list.

## 2.6 Anatomy of SimpleCNN

`SimpleCNN` is the CIFAR client: `Conv2d(3,16,3,p=1)` → LeakyReLU 0.1 → MaxPool 2 → `Conv2d(16,32,3,p=1)` → LeakyReLU → MaxPool 2 → Flatten → `FullyConnected(32*(H/4)*(W/4), C)`. Softmax is **not** in the trainable list; `forward_logits` stops at the linear layer so `CrossEntropyLoss` can consume logits, while `forward` applies Softmax for accuracy.

This topology is useless to YOLO and must not sit in `libDeepLearnLib`. It compiles in `DeepLearnModels` beside YOLO because both are thesis applications, not because they share mathematics.

## 2.7 Torch baselines are a third axis

LibTorch YOLO (`torch_baseline/TorchYOLO.hpp`, `YOLOv1` module) is neither core nor `DeepLearnModels`. It is a **measurement instrument**. Custom and Torch binaries share dataset paths via `config/experiments.json` (`voc_custom` vs `voc_torch`) and write sibling CSVs (`metrics_custom.csv`, `metrics_torch.csv`). Mixing Torch modules into `DeepLearnLib` would reintroduce the dependency the thesis exists to eliminate. Mixing Custom YOLO into Torch translation units would similarly destroy the A/B isolation.

## 2.8 Compilation footprint: what actually shrinks

“Reduce the compiled library size” has three measurable meanings:

1. **Archive size of `DeepLearnLib`.** Removing YOLO and SimpleCNN object files removes their host code, their log strings, and any unique kernel instantiations they caused. YOLO itself launched no unique kernels; it only constructed `FusedCBR2d`. The archive shrinks by the constructor and forward loop, which is modest in bytes but large in *conceptual* surface.

2. **Incremental build time of the core.** This is the dominant engineering win. Kernel files (`Tensor.cpp`, `FusedCBR2d.cpp`, `YOLOLoss.cpp`) are the slow nvcc units. Topology edits no longer dirty them.

3. **Link time of non-vision tests.** Tabular tests and tensor tests still link the full core (layers are in the same archive). They no longer need to wait on a YOLO.obj rebuild after a head-layer tweak.

The thesis should report (1) as a secondary metric and (2) as the primary developer-experience metric.

## 2.10 CMake graph after the split

The relevant fragment of the root `CMakeLists.txt` after `add_subdirectory(src)` is:

```cmake
add_library(DeepLearnModels STATIC
  benchmarks/models/YOLO.cpp
  benchmarks/models/SimpleCNN.cpp)
target_include_directories(DeepLearnModels PUBLIC
  "${CMAKE_SOURCE_DIR}/benchmarks/models")
target_link_libraries(DeepLearnModels PUBLIC DeepLearnLib)
set_target_properties(DeepLearnModels PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
```

`YOLO.cpp` is **not** in `DEEPLEARN_CUDA_SOURCES`. It contains no `__global__` kernels. Compiling it as CXX rather than CUDA is itself a footprint win: nvcc is not invoked for topology edits. The model translation units include `FusedCBR2d.hpp` and therefore *see* CUDA types (`cudaStream_t`), but they do not instantiate device code.

`src/CMakeLists.txt` remains the inventory of the generic engine. At the time of writing, `DEEPLEARN_SOURCES` is:

```text
Tensor.cpp Conv2d.cpp MaxPool2d.cpp BatchNorm2d.cpp LeakyReLU.cpp
Dropout.cpp FullyConnected.cpp Flatten.cpp Network.cpp YOLOLoss.cpp
Profiler.cpp mAP.cpp Losses.cpp Softmax.cpp CSVLoader.cpp Logger.cpp
Precision.cpp FusedCBR2d.cpp
```

plus, if OpenCV is found, `dataset.cpp`, `utils.cpp`, `ClassificationLoader.cpp`.

`DEEPLEARN_CUDA_SOURCES` is the subset that nvcc must compile. Flatten, Logger, CSVLoader, mAP, and the OpenCV loaders stay CXX. A future contributor who adds a `__global__` to `dataset.cpp` must add that file to the CUDA list or the kernel will not device-link.

Ninja’s incremental behaviour after the split:

```text
edit benchmarks/models/YOLO.cpp
  → compile YOLO.cpp.o
  → archive DeepLearnModels
  → relink train_voc_custom, inference_voc_custom, dllib_tests, …
  → do NOT compile Tensor.cpp, FusedCBR2d.cpp, YOLOLoss.cpp
```

Before the split, the same edit was a member of `DeepLearnLib`, so every consumer of the archive was downstream of a core relink, and in some generators the CUDA objects were considered dirty because they shared the library target. The measured developer-experience win is “topology tweak in seconds, kernel tweak in minutes,” which is the correct priority during thesis writing: the kernels are supposed to be stable; the detector head is not.

## 2.11 Full YOLOv1 topology as assembled from generic layers

The constructor in `benchmarks/models/YOLO.cpp` is the entire architecture. It does not implement convolution. It pushes `std::shared_ptr<Layer>`:

| Stage | Layers | Notes |
| --- | --- | --- |
| Stem | `FusedCBR2d(3,64,7,s=2,p=3)`, `MaxPool2d(2,2)` | 448→224→112 spatial (pool after stride-2 conv) |
| | `FusedCBR2d(64,192,3,p=1)`, `MaxPool2d(2,2)` | 112→56 |
| | 1×1/3×3 pair to 512, `MaxPool2d` | 56→28 |
| Mid | four × `(FusedCBR2d 512→256 1×1, 256→512 3×3)` | Redmon-style bottleneck |
| | 1×1 to 512, 3×3 to 1024, `MaxPool2d` | 28→14 |
| Deep | two × `(1024→512 1×1, 512→1024 3×3)` | |
| | 3×3 1024, 3×3 stride 2 1024, two 3×3 1024 | 14→7 grid |
| Head | `Flatten`, `FullyConnected(7*7*1024, 4096)`, `LeakyReLU(0.1)`, `Dropout(0.5)`, `FullyConnected(4096, 7*7*(10+C))` | detection tensor |

`FusedCBR2d` is Convolution + Bias + BatchNorm affine + LeakyReLU (Chapter 4). The leaky slope is 0.1, matching YOLOv1. FullyConnected layers use momentum `0.9F` in their constructors (Nesterov-style inertia on `dW` via `matmul_into` beta, Chapter 3).

`YOLO::forward` is a loop, not a graph compiler:

```cpp
const dl::StreamGuard stream_guard(stream);
dl::bind_cudnn_stream(stream);
dl::Tensor current = input_tensor.view(input_tensor.get_shape());
for (auto& layer : backbone_layers)
{
    current = layer->forward(current, stream);
    current = current.view(current.get_shape());
}
for (auto& layer : head_layers)
{
    current = layer->forward(current, stream);
    current = current.view(current.get_shape());
}
return current;
```

`StreamGuard` plus `bind_cudnn_stream` is required so that a training loop which passes a double-buffer stream (Chapter 5) actually causes cuDNN and cuBLAS to enqueue on that stream. Forgetting either call silently falls back to the default stream and destroys overlap.

`view` after each layer re-wraps the cache pointer. It is not a D2D copy when the tensor is contiguous and the element count matches. The call exists to keep the tensor’s shape metadata in sync if a layer returned a cache with a stale wrapper; it is cheap on the host.

`get_all_layers()` concatenates backbone and head so `Network` can run a uniform reverse-mode loop without knowing YOLO. Serialisation (`Network::save` / `load`) walks that flattened list. A checkpoint is therefore a sequence of generic layer blobs, not a YOLO-specific file format.

## 2.12 SimpleCNN as a second client of the same core

`SimpleCNN` exists so that the claim “the core is generic” is not only YOLO-shaped. The CIFAR-10 topology is deliberately small:

```text
Conv2d(3, 16, 3, padding=1) → LeakyReLU(0.1) → MaxPool2d(2)
Conv2d(16, 32, 3, padding=1) → LeakyReLU(0.1) → MaxPool2d(2)
Flatten → FullyConnected(32*(H/4)*(W/4), num_classes)
```

Softmax is **not** a trainable layer in the vector that `step()` walks. `forward_logits` stops at the linear layer so `CrossEntropyLoss` can consume logits (numerically stable). `forward` applies Softmax for accuracy. That split is an application concern: a detection model never calls Softmax on the 7×7 grid.

`Conv2d` here is the *unfused* core convolution, not `FusedCBR2d`. CIFAR’s spatial size (32×32) does not justify the fused BN path that YOLO’s 448×448 backbone needs; using the simpler layers also demonstrates that YOLO’s fused block is optional, not a mandatory backbone primitive.

## 2.13 `Network` versus application loops

`Network` is a convenience for detection-shaped training: it holds a `vector<shared_ptr<Layer>>`, implements `fit` with `YOLOLoss`, clips loss gradients, and writes binary checkpoints. Classification binaries (`train_cifar_custom`) do **not** call `Network::fit`. They instantiate `SimpleCNN`, call `CrossEntropyLoss`, and run their own epoch loop with `for_each_prefetched_batch`.

This is an acknowledged impurity: a textbook framework would inject a `Loss` strategy into `Network`. It is still not a topology leak. `Network.cpp` contains no `7*7*1024` constant and no `FusedCBR2d` construction. The constant `7` lives in `YOLO.cpp` and in `YOLOLoss` (the latter because the loss *is* defined on a 7×7 grid, which is a property of the loss function as used in this thesis, not of a backbone).

Tabular training is a third client: `CSVLoader` + a small stack of `FullyConnected` + `CrossEntropyLoss` or MSE, depending on the binary. Those layers are 100% core. No `DeepLearnModels` types are required for tabular except that the test binary still links the models archive for YOLO tests in the same `dllib_tests` executable.

## 2.14 Include graph and the forbidden edges

Allowed:

```text
benchmarks/train_voc_custom.cpp  →  YOLO.hpp  →  FusedCBR2d.hpp  →  Tensor.hpp
tests/test_yolo.cpp              →  YOLO.hpp
tests/test_tensor.cpp            →  Tensor.hpp   (no YOLO.hpp)
src/FusedCBR2d.cpp               →  FusedCBR2d.hpp, Tensor.hpp, Conv2d.hpp
```

Forbidden:

```text
src/*.cpp                        →  YOLO.hpp or SimpleCNN.hpp
include/DeepLearnLib/*.hpp       →  benchmarks/models/*
```

The public model include path is `benchmarks/models/`, so application code writes `#include "YOLO.hpp"`. The historical path `DeepLearnLib/YOLO.hpp` does not exist; any remaining include of that form is a compile error and a layering violation.

`dllib_configure_bench_target` adds `benchmarks/models` to private includes of every benchmark binary. That is convenience, not a license to skip linking `DeepLearnModels`. Forgetting the link fails at the YOLO constructor (undefined reference), which is the correct failure.

## 2.15 Torch baselines as a third axis

LibTorch YOLO (`torch_baseline/TorchYOLO.hpp`, `YOLOv1` module) is neither core nor `DeepLearnModels`. It is a **measurement instrument**. Custom and Torch binaries share dataset paths via `config/experiments.json` (`voc_custom` vs `voc_torch`) and write sibling CSVs (`metrics_custom.csv`, `metrics_torch.csv`). Mixing Torch modules into `DeepLearnLib` would reintroduce the dependency the thesis exists to eliminate. Mixing Custom YOLO into Torch translation units would similarly destroy A/B isolation.

The naming convention `<action>_<dataset>_<framework>` (`train_voc_custom`, `train_voc_torch`, `inference_bccd_custom`, …) exists so that an examiner can see, from the target list alone, that every scenario has a pair. Helpers `benchmarks/run_metrics.hpp`, `image_inference.hpp`, and `tabular_common.hpp` are shared *host* code for logging and I/O; they must not pull `torch::` into Custom binaries. Torch files include Torch headers only in `*_torch.cpp` and `torch_baseline/`.

## 2.16 Compilation footprint: three measurable meanings

“Reduce the compiled library size” has three meanings, and they must not be conflated in the thesis:

1. **Archive size of `DeepLearnLib`.** Removing YOLO and SimpleCNN object files removes their host code, log strings, and any unique kernel instantiations they caused. YOLO itself launched no unique kernels; it only constructed `FusedCBR2d`. The archive shrinks by the constructor and forward loop, which is modest in bytes but large in *conceptual* surface. Report this as a secondary metric.

2. **Incremental build time of the core.** This is the dominant engineering win. Kernel files (`Tensor.cpp`, `FusedCBR2d.cpp`, `YOLOLoss.cpp`) are the slow nvcc units. Topology edits no longer dirty them. Report this as the primary developer-experience metric (Ninja timestamps before/after the move).

3. **Link time of non-vision tests.** Tabular tests and tensor tests still link the full core (layers are in the same archive). They no longer wait on a YOLO.obj rebuild after a head-layer tweak. `dllib_tests` still links `DeepLearnModels` because `test_yolo.cpp` lives in the same executable; splitting the test binary would shrink link further but would complicate CI. That split is future work, not a current claim.

Static linking plus `--gc-sections` can drop unused *functions*, but it cannot drop **cuDNN algorithm heuristics and static workspaces** that constructors of unused types pull in if those types are referenced from a unity translation unit. More importantly, CMake would still *compile* `YOLO.cpp` on every core rebuild if it lived in `DEEPLEARN_SOURCES`. Compilation, not the final `.a` byte count, was the blast radius.

## 2.17 Header documentation and Doxygen

`Tensor.hpp`, `Layer.hpp`, and `Network.hpp` carry Doxygen comments that restate the contract this chapter and Chapter 3 impose: layers `ensure` caches, prefer in-place updates, and do not allocate on the hot path. Those comments are part of the public API of the *framework*, not of YOLO. Generating HTML with Doxygen (Graphviz is in the Docker image) is optional for the thesis PDF; the Markdown chapters are the narrative.

## 2.18 Dependency rule for future work

If a proposed type can be described without naming a dataset or a paper architecture, it belongs in `include/DeepLearnLib/` and `src/`. If it wires existing layers into a DAG that matches a named model, it belongs in `benchmarks/models/` (or a future `apps/`). If it exists only to compare against PyTorch, it belongs in `torch_baseline/` or a `*_torch.cpp` file.

Examples:

| Proposal | Location |
| --- | --- |
| `ConvTranspose2d` | core (`src/`, `include/DeepLearnLib/`) |
| YOLOv2 pass-through topology | `benchmarks/models/` |
| Focal loss on a grid | core *if* it is dataset-agnostic; else application |
| `torch::nn::BatchNorm2d` wrapper | `torch_baseline/` only |
| U-Net for a medical dataset | `benchmarks/models/` |

Violating this rule recreates the state this chapter documents as the *problem*: a “generic” library that is secretly a YOLO codebase.
