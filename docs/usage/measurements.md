# Measurements

`results/` is gitignored. This page is the copy that can go into the thesis. Refresh it after a local GPU run:

```bash
python scripts/freeze_measurements.py
```

GitHub-hosted runners have no NVIDIA GPU. CI compiles the training binaries and does not execute them, so this table stays empty until a machine with a GPU has written `results/**/metrics_*.csv`. The script copies the last row of each file. It does not fill in missing runs.

Each table keeps the header of that CSV. Current training runs use `Epoch;LR;TrainLoss;TestLoss;TrainAcc;TestAcc;Time(s);VRAM_MiB`, and `Time(s)` has six digits after the decimal. Forward runs use `Model;Batch;Repeats;Time(s);SamplesPerSec;VRAM_MiB`. Rows already on this page stay as they were until those binaries are run again.

## Recorded runs

### `results/cifar10/metrics_custom.csv`

| File | Epoch | TrainLoss | TestLoss | Time(s) | VRAM_MiB | TrainAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| results/cifar10/metrics_custom.csv | 40 | 2.41182 | 1.39182 | 53 | 1521 | 0.732577 | 0.656052 |

### `results/cifar10/metrics_torch.csv`

| File | Epoch | TrainLoss | TestLoss | Time(s) | VRAM_MiB | TrainAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| results/cifar10/metrics_torch.csv | 40 | 0.575592 | 0.900853 | 53 | 1601 | 0.807101 | 0.700455 |

### `results/mnist/metrics_custom.csv`

| File | Epoch | TrainLoss | TestLoss | Time(s) | VRAM_MiB | TrainAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| results/mnist/metrics_custom.csv | 12 | 2.79275 | 5.50321 | 1 | 1515 | 0.990261 | 0.986353 |

### `results/mnist/metrics_torch.csv`

| File | Epoch | TrainLoss | TestLoss | Time(s) | VRAM_MiB | TrainAcc | TestAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| results/mnist/metrics_torch.csv | 12 | 0.0327417 | 0.0368073 | 1 | 1601 | 0.990422 | 0.987935 |

### `results/tabular/metrics_custom.csv`

| File | Epoch | Loss | Time(s) | VRAM_MiB | Acc |
| --- | --- | --- | --- | --- | --- |
| results/tabular/metrics_custom.csv | 40 | 0.00445682 | 0 | 1481 | 1 |

### `results/tabular/metrics_torch.csv`

| File | Epoch | Loss | Time(s) | VRAM_MiB | Acc |
| --- | --- | --- | --- | --- | --- |
| results/tabular/metrics_torch.csv | 40 | 0.00504049 | 0 | 1553 | 1 |

### `results/tabular_iris/metrics_custom.csv`

| File | Epoch | Loss | Time(s) | VRAM_MiB | Acc |
| --- | --- | --- | --- | --- | --- |
| results/tabular_iris/metrics_custom.csv | 100 | 0.0451643 | 0 | 1481 | 0.986667 |

### `results/tabular_iris/metrics_torch.csv`

| File | Epoch | Loss | Time(s) | VRAM_MiB | Acc |
| --- | --- | --- | --- | --- | --- |
| results/tabular_iris/metrics_torch.csv | 100 | 0.048417 | 0 | 1553 | 0.98 |

### `results/tabular_wisconsin/metrics_custom.csv`

| File | Epoch | Loss | Time(s) | VRAM_MiB | Acc |
| --- | --- | --- | --- | --- | --- |
| results/tabular_wisconsin/metrics_custom.csv | 80 | 0.00633334 | 0 | 1481 | 1 |

### `results/tabular_wisconsin/metrics_torch.csv`

| File | Epoch | Loss | Time(s) | VRAM_MiB | Acc |
| --- | --- | --- | --- | --- | --- |
| results/tabular_wisconsin/metrics_torch.csv | 80 | 0.00702566 | 0 | 1553 | 1 |
