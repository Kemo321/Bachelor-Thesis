# Running

Compile flags, the Ninja generator, and SASS pinning are in [library build](../library/build.md). This page is how a built tree is driven.

## Docker

```bash
docker compose up -d --build
docker exec -it yolo_dev_container bash
./scripts/dev.sh
./scripts/menu.sh
```

The image is an NGC PyTorch container because that tag pins CUDA, cuDNN, and a Python stack together. `DeepLearnLib` still does not include LibTorch. Compose mounts the repository at `/app`, keeps `build/` and ccache in named volumes, sets `/dev/shm` to 32 GiB, and reserves one GPU. The default 64 MiB of `/dev/shm` is too small for the decode threads.

`dev.sh` configures with Ninja and `USE_CUDA=ON`, uses ccache when it is on `PATH`, then builds and runs `dllib_tests`.

## Menu

`scripts/menu.sh` groups entries by dataset, then inference, Google Benchmark targets, and plots. Each Custom scenario has a Torch twin when that target was generated.

The sanity entry points `EXPERIMENTS_JSON` at `config/sanity.json` and runs the same binaries with short epoch counts. It checks that a rebuild still steps on the GPU. Thesis numbers come from `config/experiments.json`.

## Local GPU check

`scripts/verify_gpu.sh` and `scripts/verify_gpu.ps1` run the built `dllib_tests` after `nvidia-smi` succeeds. `--short` also runs `short_voc_custom` (3 epochs) when `data/VOCdevkit/VOC2012` is present, and skips that binary when the data is absent. Both scripts exit immediately when `GITHUB_ACTIONS` is set. They are not wired into the workflow. Hosted runners have no NVIDIA GPU, so CI compiles and does not execute these binaries.

## Plots

```bash
python3 scripts/plot_metrics.py --results-root results
```

The script reads the CSVs described in [Experiments](experiments.md). It does not train.

## Windows

`powershell -File scripts/dev.ps1` loads the Visual C++ environment and then runs Git Bash. Configure from that shell. WSL `bash.exe` does not see MSVC's include path, and nvcc's host compile then fails to find the C standard library.
