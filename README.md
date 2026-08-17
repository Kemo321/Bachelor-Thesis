# DeepLearnLib

Custom C++/CUDA deep-learning library written from scratch for a Bachelor's thesis. The core (`DeepLearnLib`) does **not** depend on LibTorch: tensors, layers, and the optimiser are implemented on NVIDIA's stack (CUDA, cuBLAS, cuDNN, Thrust).

The design is intentionally **lightweight**. Application networks (YOLOv1, a CIFAR-10 CNN) live outside the compiled core and are assembled from generic layers. The training path prefers **static VRAM caches** over per-step `cudaMalloc`, trading memory for LibTorch-like compute time.

## Abstract

DeepLearnLib provides a dense `dl::Tensor` with GPU storage, in-place arithmetic, and zero-allocation GEMM (`matmul_into`). Convolution, pooling, and batch-norm use the cuDNN C API; fully-connected layers use `cublasSgemm` with logical transposes. Elementwise kernels (LeakyReLU, Dropout, fused SGD) stay on device. Host synchronisation is reserved for logging, metrics, and checkpoints.

## Architecture overview

```
┌─────────────────────────────────────────────┐
│  Applications (benchmarks/, tests/)         │
│  YOLO, SimpleCNN, training/inference CLIs   │
└─────────────────────┬───────────────────────┘
                      │ compose layers
┌─────────────────────▼───────────────────────┐
│  DeepLearnLib (src/, include/DeepLearnLib/) │
│  Tensor, Layer, Conv2d, FC, FusedCBR2d, …   │
│  cuDNN / cuBLAS / CUDA kernels              │
└─────────────────────────────────────────────┘
```

- **Core library** — generic tensors, layers, loaders, and losses. No YOLO or CIFAR topology is linked into `libDeepLearnLib`.
- **Models** — `benchmarks/models/YOLO.{hpp,cpp}` and `SimpleCNN.{hpp,cpp}` are compiled into `DeepLearnModels` and linked only by apps and tests.
- **Torch baselines** — optional LibTorch binaries (`*_torch`) for apples-to-apples timing. They are not part of the custom stack.

See the numbered thesis chapters in [docs/README.md](docs/README.md). Start with [Chapter 1 — Introduction and setup](docs/01_INTRODUCTION_AND_SETUP.md). How to add a generic layer: [docs/ADDING_LAYERS.md](docs/ADDING_LAYERS.md).

## Prerequisites

- NVIDIA GPU with a recent driver (CUDA 12+ toolkit)
- [Docker](https://docs.docker.com/get-docker/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) **or** a local toolchain:
  - CMake 3.18+
  - Ninja
  - CUDA 12+ (nvcc, cuBLAS, cuDNN)
  - OpenCV, pugixml, a C++17 compiler

LibTorch is optional and used only for baseline binaries.

## Installation

### Docker (recommended)

From the repository root:

```bash
docker compose up -d --build
docker exec -it yolo_dev_container bash
./scripts/dev.sh          # configure (Ninja), build, run unit tests
./scripts/menu.sh         # interactive training / inference / plots
```

The compose file mounts the repo at `/app`, persists `build/` and ccache, and requests one NVIDIA GPU.

### Local build

```bash
cmake -S . -B build -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/tests/dllib_tests
```

On Windows use `scripts/dev.ps1` (loads MSVC + Git Bash) rather than WSL `bash.exe`.

## Usage

### Interactive menu

```bash
./scripts/menu.sh
```

Options are grouped by dataset (VOC, BCCD, Synthetic, CIFAR-10, Tabular), then inference, Google Benchmark targets, and plots. Each Custom scenario has a matching Torch binary (`train_voc_custom` / `train_voc_torch`, …).

### Sanity check

Menu item **Sanity Check** sets `EXPERIMENTS_JSON` to `config/sanity.json` (2-epoch smoke configs) and runs the same pipeline sequence. Useful after a rebuild to prove the stack still links and steps on GPU.

### Metrics

Training binaries write `results/<experiment>/metrics_custom.csv` and `metrics_torch.csv` (`Epoch`, `Loss` / train–test losses, `Time(s)`, `VRAM_MiB`). Plot with:

```bash
python3 scripts/plot_metrics.py --results-root results
```

Details: [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Adding a layer: [docs/ADDING_LAYERS.md](docs/ADDING_LAYERS.md).

## Project structure

```
.
├── include/DeepLearnLib/     # Public core API (Tensor, Layer, Conv2d, …)
├── src/                      # Core library implementation (no YOLO/SimpleCNN)
├── benchmarks/
│   ├── models/               # Application networks: YOLO, SimpleCNN
│   ├── train_*_custom.cpp    # Custom training CLIs
│   ├── train_*_torch.cpp     # LibTorch baselines
│   └── inference_*.cpp
├── tests/                    # GoogleTest suite (dllib_tests)
├── torch_baseline/           # Optional LibTorch YOLO / dataset
├── config/                   # experiments.json, sanity.json
├── scripts/                  # menu.sh, dev.sh, plot_metrics.py
├── docs/                     # Thesis chapters 01–06 plus ADDING_LAYERS.md
└── docker-compose.yml
```

## License

Bachelor's thesis project — see the repository for academic use.
