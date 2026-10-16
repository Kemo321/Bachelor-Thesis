# Experiments

Each dataset has a Custom binary and, when LibTorch was found at configure time, a Torch binary. Both read the same block of `config/experiments.json`. `config/sanity.json` is the same schema with short epoch counts. The two binaries share one epoch loop, one optimiser schedule, and one CSV.

## Pipelines

| JSON key | Binary | Model | Optimiser |
| --- | --- | --- | --- |
| `cifar10_classification` | `train_cifar_*` | `SimpleCNN` / `TorchSimpleCNN` | SGD |
| `mnist_classification` | `train_mnist_*` | `SimpleCNN` / `TorchSimpleCNN` | SGD |
| `tabular_demo`, `tabular_iris`, `tabular_wisconsin` | `train_tabular_*` | `TabularMLP` / `TorchTabularMLP` | SGD |
| `ecg200` | `train_gru_*` | `GruClassifier` / `TorchGruClassifier` | Adam |
| `forward_latency` | `bench_forward_*` | the four models above, eval only | — |

Every classifier uses `CrossEntropyLoss`. Tabular holds out `test_fraction` of the rows (default 0.2) with seed 42, so it reports the same test columns as the image and ECG runs. ECG200 already has a train file and a test file.

## Benchmarks

`bench_forward_*` times `forward` on a fixed input for every model in `forward_latency.cases`. Custom and Torch read that same list. `bench_micro_ops` times single kernels. Epoch time stays in the training CSV; kernel time stays in the micro-benchmark.

## CSV

Training writes `results/<experiment>/metrics_custom.csv` or `metrics_torch.csv`. The separator is a semicolon. The decimal point is `.`. There is no text log next to the CSV.

Every training program writes the same header:

```text
Epoch;LR;TrainLoss;TestLoss;TrainAcc;TestAcc;Time(s);VRAM_MiB
```

`Time(s)` is the wall time of the epoch, including the test pass, with six digits after the decimal (microseconds). Loss is the sample-weighted mean of the per-batch means. Accuracy is `correct / seen`, so a short last batch does not count as a full batch. `VRAM_MiB` is the process device-memory reading after the epoch.

Forward latency writes:

```text
Model;Batch;Repeats;Time(s);SamplesPerSec;VRAM_MiB
```

`Time(s)` there is the timed repeat loop after warmup, again with six decimal digits.

## What the two stacks share

The comparison is the same GPU, the same batch size, the same epoch count, the same schedule, the same split, the same loss reduction, the same initial weights, and the same CSV. Both stacks shuffle with seed 42 through one Fisher-Yates implementation, and image paths are sorted before that shuffle. Custom and Torch binaries are separate executables. Custom does not link LibTorch. Convolution and GEMM still run as TF32 tensor-core kernels, so a later epoch can differ in the last bits; the runs start from the same state.

CI compiles Custom binaries without a GPU. A green CI run means the tree links. Epoch time, VRAM, and agreement with the Torch binaries come from a machine that can run the cubin. Architecture pinning for that cubin is in [Build](../library/build.md). The numbers that belong in the thesis are copied into [Measurements](measurements.md) by `scripts/freeze_measurements.py` after that local run. The script does not invent rows when `results/` is empty.
