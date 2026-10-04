# Experiments

Each dataset has a Custom binary and, when LibTorch was found at configure time, a Torch binary. Both read a block of the same shape from `config/experiments.json` (batch size, epochs, schedule, paths). `config/sanity.json` is the same schema with short epoch counts.

## Pipelines

| JSON key | Binary | Model | Loss | Extra metric |
| --- | --- | --- | --- | --- |
| `voc_custom`, `voc_torch` | `train_voc_*` | `YOLO`, 20 classes | `YOLOLoss` | mAP@0.5 |
| `bccd_custom`, `bccd_torch` | `train_bccd_*` | `YOLO`, 3 classes | `YOLOLoss` | mAP@0.5 |
| `synthetic_custom`, `synthetic_torch` | `train_synthetic_*` | `YOLO`, 3 classes | `YOLOLoss` | mAP@0.5 |
| `cifar10_classification` | `train_cifar_*` | `SimpleCNN` | `CrossEntropyLoss` | accuracy |
| `mnist_classification` | `train_mnist_*` | `SimpleCNN` | `CrossEntropyLoss` | accuracy |
| `tabular_demo`, `tabular_iris`, `tabular_wisconsin` | `train_tabular_*` | two-layer MLP | `CrossEntropyLoss` | accuracy |
| `overfit_voc_*` | `overfit_voc_*` | `YOLO` | `YOLOLoss` | detection CSV on the tiny slice |
| short VOC | `short_voc_*` | `YOLO` | `YOLOLoss` | detection CSV, 3 epochs |

Every detection binary writes `Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;mAP@0.5`. `overfit_voc_*` computes test loss and mAP on the same images used for the step, after `eval`. `short_voc_*` uses the VOC test split. Classification writes `TrainAcc` and `TestAcc` instead of mAP. Tabular writes `Epoch;TrainLoss;Time(s);VRAM_MiB;TrainAcc` because that loader has no held-out split.

## Benchmarks

`bench_voc_custom` and `bench_voc_torch` time a VOC step with Google Benchmark.

`bench_micro_ops` times single operations (elementwise kernels, GEMM shapes used by the head, convolution, batch-norm, the fused block) on the Custom stack and, when Torch is enabled, on LibTorch. That binary is where a gap is attributed to one kernel rather than to the whole epoch.

## CSV

Training writes `results/<experiment>/metrics_custom.csv` or `metrics_torch.csv`. The separator is a semicolon. The same directory receives `log_custom.txt` or `log_torch.txt`: the info lines from the console, replaced when that program starts. Inference writes `log_infer_custom.txt` or `log_infer_torch.txt` there, so a drawing run leaves the training log in place. `logs/framework.log` stays the rotating trace log.

Detection:

```text
Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;mAP@0.5
```

Classification:

```text
Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;TrainAcc;TestAcc
```

Tabular, no held-out split:

```text
Epoch;TrainLoss;Time(s);VRAM_MiB;TrainAcc
```

`Time(s)` is wall time of the epoch, including the test pass and mAP or accuracy. It is not a kernel time. Kernel time comes from `Profiler` or from `bench_micro_ops`. `VRAM_MiB` is the process device-memory reading at the log point. On the Custom stack the buffers from `Tensor::ensure` stay allocated, so the column should be flat across epochs. A climb means a new allocation every step.

mAP is `mean_average_precision` at IoU 0.5. Boxes come from `detections_from_tensor` after the confidence threshold and NMS. The library definition of the score is in [Utilities](../library/utilities.md).

## What the two stacks share

The comparison is the same GPU, the same batch size, the same epoch count, the same schedule, and the same split. Custom and Torch binaries are separate executables. Custom does not link LibTorch.

CI compiles Custom binaries without a GPU. A green CI run means the tree links. Epoch time, VRAM, and agreement with the Torch binaries come from a machine that can run the cubin. Architecture pinning for that cubin is in [Build](../library/build.md). The numbers that belong in the thesis are copied into [Measurements](measurements.md) by `scripts/freeze_measurements.py` after that local run. The script does not invent rows when `results/` is empty.
