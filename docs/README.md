# DeepLearnLib documentation

This directory is the long-form technical record of the Bachelor's thesis framework. It is organised as numbered chapters. Read them in order; later chapters assume the vocabulary of earlier ones. The chapters are written at dissertation length: they cite CMake flags, CUDA launch parameters, and source file names so an examiner can grep the tree.

| Chapter | File | Subject |
| --- | --- | --- |
| 1 | [01_INTRODUCTION_AND_SETUP.md](01_INTRODUCTION_AND_SETUP.md) | Formal abstract; CMake, Ninja, ccache; C++17/CUDA flags; SASS vs PTX; Docker Compose; MSVC/`dev.ps1`; CI |
| 2 | [02_CORE_VS_MODELS_ARCHITECTURE.md](02_CORE_VS_MODELS_ARCHITECTURE.md) | Separation of concerns; `DeepLearnLib` vs `DeepLearnModels`; YOLOv1 and SimpleCNN as clients; include graph |
| 3 | [03_MEMORY_MANAGEMENT_AND_TENSORS.md](03_MEMORY_MANAGEMENT_AND_TENSORS.md) | `dl::Tensor`, `CudaDeleter`, `ensure`, ~13 GiB tradeoff, `matmul_into` / `CUBLAS_OP_T` |
| 4 | [04_KERNEL_FUSION_AND_COMPUTE.md](04_KERNEL_FUSION_AND_COMPUTE.md) | Custom `__global__` vs Thrust; `FusedCBR2d`; `YOLOLoss` GPU reduce |
| 5 | [05_ASYNCHRONOUS_DATALOADING.md](05_ASYNCHRONOUS_DATALOADING.md) | JPEG thread pool, `std::future` prefetch, double-buffered streams |
| 6 | [06_BENCHMARKS_AND_PROFILING.md](06_BENCHMARKS_AND_PROFILING.md) | CUDA Events, `bench_micro_ops`, CSV columns, `plot_metrics.py` |

Short indexes (historical filenames): [ARCHITECTURE.md](ARCHITECTURE.md), [BENCHMARKS.md](BENCHMARKS.md).

Contributor tutorial (how to add a generic `Layer`): [ADDING_LAYERS.md](ADDING_LAYERS.md).

Internal engineering notes: [dev/](dev/).
