# Benchmarks

Every training and evaluation scenario has a **Custom** and a **Torch** binary named `<action>_<dataset>_<framework>` (for example `train_voc_custom` / `train_voc_torch`). Models used by Custom apps live in `benchmarks/models/` and are **not** part of `DeepLearnLib`.

## Interactive menu

```bash
./scripts/dev.sh      # first-time configure + unit tests
./scripts/menu.sh
```

Groups:

- Training (VOC / BCCD / Synthetic / CIFAR-10 / Tabular)
- Overfit and short VOC smokes
- Inference
- Google Benchmark (`bench_voc_*`, `bench_micro_ops`)
- Plot metrics
- Run all / Sanity Check (`config/sanity.json`)

Missing Torch binaries are skipped (LibTorch not configured). Failures in **Run all** do not abort the rest of the sequence.

## Micro-benchmarks (`bench_micro_ops`)

Compares isolated ops (GEMM / FC, conv blocks, YOLO loss, SimpleCNN, …) on Custom tensors vs LibTorch. Requires `TorchBaseline`.

```bash
# from the build tree, or via the menu
./build/benchmarks/bench_micro_ops \
  --benchmark_min_time=0.5s \
  --benchmark_counters_tabular=true
```

Google Benchmark prints ns/op (or similar) and extra counters. Lower is better for latency. Use the same GPU clocks and batch sizes as the thesis tables. `bench_micro_more.cpp` holds additional Custom-vs-Torch cases compiled into the same target.

Interpret:

- **Custom faster or within a few percent of Torch** on FC backward / YOLO forward is expected after `matmul_into`, fused SGD, and GPU loss reduce.
- **Large Custom regressions** usually mean a sync (`to_host` in the hot loop) or a fresh `cudaMalloc` (cache miss because of changing shapes).

Epoch-level YOLO timing (full data loader) is `bench_voc_custom` / `bench_voc_torch`:

```bash
./build/benchmarks/bench_voc_custom --benchmark_min_time=0.1s
./build/benchmarks/bench_voc_torch --benchmark_min_time=0.1s
```

These need VOC under `data/VOCdevkit` (see `scripts/setup_datasets.py`).

## Training metrics CSVs

Pipelines log with `dl::Logger` and write semicolon-separated CSVs:

| Column | Meaning |
| --- | --- |
| `Epoch` | 1-based epoch index |
| `Loss` or `TrainLoss` / `TestLoss` | Scalar loss (Custom `YOLOLoss` / CE vs Torch equivalent) |
| `Time(s)` | Wall-clock epoch duration |
| `VRAM_MiB` | `Profiler::get_vram_usage_mb()` after the epoch |
| extras | `mAP@0.5`, `TrainAcc` / `TestAcc` when applicable |

Paths:

```text
results/voc/metrics_custom.csv
results/voc/metrics_torch.csv
results/bccd/...
results/synthetic/...
results/cifar10/...
results/tabular/...
results/overfit/...
results/voc_short/...
```

VRAM is **process used memory**, not a CUDA graph peak. Custom caches make this number higher than a naive allocate-free loop; that is the intended time/memory tradeoff ([ARCHITECTURE.md](ARCHITECTURE.md)).

## Plotting

```bash
python3 scripts/plot_metrics.py --results-root results
```

Requires `matplotlib` and `pandas`. For each experiment directory the script writes:

- `plots/train_vs_test_loss.png`
- `plots/map50.png` (skipped if no mAP column)
- `plots/epoch_duration.png`
- `plots/vram.png`

and copies figures into `results/plots/`. Column names are normalised (`Loss` → train loss, `VRAM_MiB`, …). Missing Torch CSVs plot Custom only.

## Sanity Check

Menu **Sanity Check** exports `EXPERIMENTS_JSON=config/sanity.json` (2 epochs, small batches) and runs tabular + the full `run_all` sequence. Use it to verify binaries after CMake changes, not as a quality metric.
