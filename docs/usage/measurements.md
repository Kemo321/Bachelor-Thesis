# Measurements

This page lists the last row of each CSV file in `results/`, which holds the measurements quoted in the thesis. Refresh it after a local GPU run:

```bash
python scripts/freeze_measurements.py
```

GitHub-hosted runners have no NVIDIA GPU. CI compiles the training binaries and does not execute them, so this table stays empty until a machine with a GPU has written `results/**/metrics_*.csv`. The script copies the last row of each file. It does not fill in missing runs.

Each table keeps the header of that CSV. Current training runs use `Epoch;LR;TrainLoss;TestLoss;TrainAcc;TestAcc;Time(s);VRAM_MiB`, and `Time(s)` has six digits after the decimal. Forward runs use `Model;Batch;Repeats;Time(s);SamplesPerSec;VRAM_MiB`. Rows already on this page stay as they were until those binaries are run again.

## Recorded runs

### `results/cifar10/metrics_custom.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/cifar10/metrics_custom.csv | 40 | 0.000100 | 0.577741 | 0.887538 | 0.804920 | 0.704100 | 39.385273 | 1549 |

### `results/cifar10/metrics_torch.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/cifar10/metrics_torch.csv | 40 | 0.000100 | 0.576996 | 0.888140 | 0.805340 | 0.702300 | 39.801740 | 1603 |

### `results/ecg200/metrics_custom.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/ecg200/metrics_custom.csv | 80 | 0.001000 | 0.533121 | 0.616284 | 0.730000 | 0.640000 | 0.009365 | 1523 |

### `results/ecg200/metrics_torch.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/ecg200/metrics_torch.csv | 80 | 0.001000 | 0.533121 | 0.616284 | 0.730000 | 0.640000 | 0.029114 | 1583 |

### `results/forward/metrics_custom.csv`

| File | Model | Batch | Repeats | Time(s) | SamplesPerSec | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- |
| results/forward/metrics_custom.csv | gru_classifier | 32 | 30 | 0.002954 | 324944.683245 | 1483 |

### `results/forward/metrics_torch.csv`

| File | Model | Batch | Repeats | Time(s) | SamplesPerSec | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- |
| results/forward/metrics_torch.csv | gru_classifier | 32 | 30 | 0.016082 | 59692.635160 | 1517 |

### `results/mnist/metrics_custom.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/mnist/metrics_custom.csv | 12 | 0.000100 | 0.033662 | 0.038390 | 0.990150 | 0.986400 | 1.231049 | 1541 |

### `results/mnist/metrics_torch.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/mnist/metrics_torch.csv | 12 | 0.000100 | 0.033744 | 0.038398 | 0.990183 | 0.986400 | 1.295291 | 1605 |

### `results/tabular/metrics_custom.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tabular/metrics_custom.csv | 40 | 0.000500 | 0.004550 | 0.003906 | 1.000000 | 1.000000 | 0.005510 | 1481 |

### `results/tabular/metrics_torch.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tabular/metrics_torch.csv | 40 | 0.000500 | 0.004550 | 0.003905 | 1.000000 | 1.000000 | 0.004133 | 1557 |

### `results/tabular_iris/metrics_custom.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tabular_iris/metrics_custom.csv | 100 | 0.000500 | 0.021818 | 0.134768 | 1.000000 | 0.966667 | 0.005748 | 1481 |

### `results/tabular_iris/metrics_torch.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tabular_iris/metrics_torch.csv | 100 | 0.000500 | 0.021808 | 0.134847 | 1.000000 | 0.966667 | 0.008113 | 1557 |

### `results/tabular_wisconsin/metrics_custom.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tabular_wisconsin/metrics_custom.csv | 80 | 0.000500 | 0.005826 | 0.250229 | 1.000000 | 0.973684 | 0.022079 | 1481 |

### `results/tabular_wisconsin/metrics_torch.csv`

| File | Epoch | LR | TrainLoss | TestLoss | TrainAcc | TestAcc | Time(s) | VRAM_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/tabular_wisconsin/metrics_torch.csv | 80 | 0.000500 | 0.005822 | 0.250364 | 1.000000 | 0.973684 | 0.017184 | 1557 |

