# Chapter 6 — Benchmarks and profiling

## 6.1 Measurement philosophy

A from-scratch library that “feels fast” is not a result. The thesis requires:

1. **Micro-benchmarks** that isolate one operator (GEMM, conv, YOLO loss, SGD) on Custom versus LibTorch, using CUDA Events rather than `std::chrono` around a kernel launch (the latter includes host scheduling noise and misses GPU-side asynchrony).
2. **Epoch-level benchmarks** that include the dataloader (`bench_voc_custom` / `bench_voc_torch`).
3. **Full pipelines** that write CSVs (`Epoch`, losses, `Time(s)`, `VRAM_MiB`) for plots used in the dissertation.
4. **Honesty about VRAM.** Custom's ~13 GiB YOLO footprint is reported, not hidden. `Profiler::get_vram_usage_mb` is `cudaMemGetInfo` used memory, not an Nsight peak of a single kernel.

Google Benchmark (`benchmark::State`) is the harness for (1) and (2). `scripts/plot_metrics.py` is the harness for (3).

## 6.2 `Profiler`: CUDA Events and VRAM

```cpp
class Profiler {
    void start();           // cudaEventRecord(start) on current stream
    float stop();           // record stop, synchronize stop event, elapsed ms
    static size_t get_vram_usage_mb();
};
```

`start()` does **not** synchronise. It stamps the stream. `stop()` is the first host wait for that interval. This matches how Nsight and LibTorch's `torch.cuda.Event` measure GPU time.

`get_vram_usage_mb` implements:

```cpp
size_t free_bytes, total_bytes;
cudaMemGetInfo(&free_bytes, &total_bytes);
return (total_bytes - free_bytes) / (1024 * 1024);
```

Interpretation caveats for the thesis text:

- The number includes **all** contexts on the device that the runtime attributes to the process, plus allocator caches.
- It is **not** “how much YOLO needs in theory.” It is “what the driver reports after caches are warm.”
- Comparing Custom 13 GiB to Torch 11 GiB (illustrative) is valid; comparing Custom 13 GiB to a handwritten formula of weight bytes only is not.

## 6.3 `bench_micro_ops` suite

File: `benchmarks/bench_micro_ops.cpp` plus `bench_micro_more.cpp`, one CMake target `bench_micro_ops`, linked with `DeepLearnLib`, `DeepLearnModels`, and `TorchBaseline`. CUDA is required (`require_cuda` skips otherwise).

Constants (must match thesis tables unless the code is changed):

```text
kBatch     = 16
kImage     = 448
kFcIn      = 7*7*1024
kFcOut     = 4096
kWarmup    = 5
```

Warmup matters: the first cuDNN `find` and the first `ensure` allocate. Timing those would credit LibTorch's already-warm context and punish Custom. Each Custom fixture runs `kWarmup` forwards (and backwards where relevant) before `benchmark::State` iterations.

### 6.3.1 Timing loop

```cpp
template <typename Body>
void run_gpu_loop(benchmark::State& state, size_t bytes, Body&& body)
{
    Profiler profiler;
    for (auto _ : state) {
        profiler.start();
        body();
        float ms = profiler.stop();
        state.SetIterationTime(ms / 1000.0);
    }
    state.SetBytesProcessed(iterations * bytes);
    state.counters["VRAM_MiB"] = (double)Profiler::get_vram_usage_mb();
}
```

`SetIterationTime` tells Google Benchmark to use **measured GPU seconds**, not wall time of the `for` loop. `--benchmark_counters_tabular=true` prints `VRAM_MiB` as a column.

### 6.3.2 What is compared

The suite (including `bench_micro_more.cpp`) is intended to cover:

- Custom `Conv2d` versus `torch::nn::Conv2d` at YOLO-like spatial size (`kSpatial = 112`, 64→192, 3×3);
- Custom `FullyConnected` forward/backward versus `torch::nn::Linear` at head dimensions;
- `YOLOLoss` versus `compute_yolo_loss`;
- in-place Custom elementwise versus Torch ops;
- SimpleCNN / YOLO module forwards where those fixtures exist.

When reading a result, check:

- **items_per_second / bytes_per_second** if reported;
- **manual time** in ns or ms (GPU);
- **VRAM_MiB** after the loop (Custom higher is expected).

A Custom kernel that is 2× slower with equal VRAM is a compute problem (fusion, transpose copies). A Custom kernel that is equal time with much higher VRAM is the Chapter 3 tradeoff working as designed. A Custom kernel that is slower *and* lower VRAM often means it is still allocating per iteration—warmup was skipped or `ensure` is missing.

### 6.3.3 How to run

From the menu: **Micro-benchmarks**. From the shell:

```bash
./scripts/menu.sh
# or
ninja -C build bench_micro_ops
./build/benchmarks/bench_micro_ops \
  --benchmark_min_time=0.5s \
  --benchmark_counters_tabular=true \
  --benchmark_filter=FC   # optional regex
```

`--benchmark_min_time=0.5s` keeps GPU clocks from sitting in a 1-iteration turbo/idle mess. For thesis tables, fix clocks if the OEM allows (`nvidia-smi -lgc`), disable boosting, and average several process launches.

## 6.4 Epoch-level Google Benchmarks

`bench_voc_custom` / `bench_voc_torch` run a full YOLO training step over the VOC loader (or skip if `data/VOCdevkit` is empty). They measure *including* decode and H2D. They are the right tool when Chapter 5 claims overlap; micro-ops will not show a dataloader win.

```bash
./build/benchmarks/bench_voc_custom --benchmark_min_time=0.1s
./build/benchmarks/bench_voc_torch  --benchmark_min_time=0.1s
```

## 6.5 Full training CSVs

Every `train_*` binary uses `dl::Logger` and writes a semicolon-separated file:

```text
results/<experiment>/metrics_custom.csv
results/<experiment>/metrics_torch.csv
```

Minimum columns (Chapter 2 naming):

```text
Epoch ; Loss or TrainLoss/TestLoss ; Time(s) ; VRAM_MiB
```

Optional: `mAP@0.5`, `TrainAcc`, `TestAcc`.

Example VOC Custom header:

```text
Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;mAP@0.5
1;142.3;140.1;12;13100;0.002
```

`Time(s)` is **host** `std::chrono::steady_clock` around the whole epoch (train + eval + logging). It is not a CUDA Event. That is intentional: the thesis question for plots is “how long until the next epoch,” which includes JPEG and `to_host` for mAP. Micro-ops answer “how long is the GEMM.”

`VRAM_MiB` is sampled once per epoch after the step (`Profiler::get_vram_usage_mb()`). It should be almost constant after epoch 1 if `ensure` works. A rising curve indicates a leak (missing `shared_ptr` aliasing, or Torch caching). A drop on eval-only epochs would mean caches were freed; Custom currently keeps them.

Experiments:

| Directory | Produced by |
| --- | --- |
| `results/voc/` | `train_voc_custom`, `train_voc_torch` |
| `results/bccd/` | BCCD pair |
| `results/synthetic/` | Synthetic pair |
| `results/cifar10/` | `train_cifar_*` |
| `results/tabular/` | `train_tabular_*` |
| `results/overfit/` | `overfit_voc_*` |
| `results/voc_short/` | `short_voc_*` |

## 6.6 `plot_metrics.py`

```bash
python3 scripts/plot_metrics.py --results-root results
```

Requires matplotlib and pandas. For each experiment directory the script:

1. loads `metrics_custom.csv` and `metrics_torch.csv` (semicolon, fallback comma);
2. normalises column names (`Loss` → `TrainLoss`, `VRAM_MiB`, `Time(s)`, `mAP@0.5`);
3. ignores accuracy columns so they cannot collide with loss;
4. writes `plots/train_vs_test_loss.png`, `map50.png` (if present), `epoch_duration.png`, `vram.png`;
5. copies figures to `results/plots/<experiment>_*.png`.

**How to read the figures in the thesis:**

- **Train vs test loss:** Custom and Torch should decrease on a similar schedule if losses are the same function. A Custom curve that is a scaled copy of Torch suggests a missing `1/B` in `YOLOLoss`. A Custom curve that diverges after epoch 5 suggests LR schedule mismatch (`scheduled_learning_rate` vs Torch `lr_schedule` in JSON).
- **Epoch duration:** Custom slightly above Torch is the residual 1–2 ms/batch × batches. A Custom line twice Torch is a sync or copy regression; re-run `bench_micro_ops` to localise.
- **VRAM:** Custom flat and high; Torch may sit lower. Annotate the plot with the tradeoff sentence from Chapter 3.
- **mAP:** only VOC Custom currently writes mAP; Torch VOC training CSV may omit it. Do not claim a detection-quality win without the Torch mAP column.

## 6.7 Interactive menu and sanity

`scripts/menu.sh` builds targets with Ninja, runs them from the binary directory, and groups Custom/Torch pairs. **Run all** skips missing Torch binaries. **Sanity Check** exports `EXPERIMENTS_JSON=config/sanity.json` (2 epochs, `dataloader_workers: 0`) so a rebuild can be smoke-tested without a 150-epoch VOC run.

Sanity is **not** a performance result. Quote only full `experiments.json` runs, or micro-ops, in the dissertation tables.

## 6.8 Suggested thesis table layout

For each operator in `bench_micro_ops`:

| Operator | Custom ms | Torch ms | Custom VRAM MiB | Torch VRAM MiB | Notes |
| --- | --- | --- | --- | --- | --- |
| FC forward | | | | | `matmul_into`, TF32 |
| FC backward | | | | | logical `OP_T` |
| Conv 3×3 | | | | | cuDNN find cached |
| YOLOLoss | | | | | device reduce |
| YOLO train step | | | ~13000 | | includes caches |

Fill from a single pinned-clock session. Record driver version, `nvidia-smi` clocks, batch size 16, and compile arch (`120-real` or whatever `cuda_env.sh` printed).

## 6.9 Nsight (optional, not required to run the suite)

`dl::NvtxRange` markers (`FullyConnected_Forward`, `DataLoader_GetBatch`, `YOLOLoss_Loss`) appear in Nsight Systems. Use them to verify Chapter 5: decode threads should be busy while `FullyConnected_Forward` is on the GPU. If `DataLoader_GetBatch` sits on the same track *before* every forward with the GPU idle, prefetch is broken.

## 6.10 What not to measure

- First iteration after process start (cuDNN find, `cudaMalloc`).
- Debug + ASan builds (host sanitizers destroy GPU timing).
- Mixed Custom/Torch in one process except `bench_micro_ops`, which is designed for that (it will show *sum* VRAM).
- `extern/pytorch` submodule rebuilds as if they were Custom performance work.

## 6.11 Reading a metrics CSV by hand

`dl::Logger` (spdlog) writes colour stdout and a rotating TRACE file. That is **not** the thesis table. Thesis tables come from semicolon-separated files written by `benchmarks/run_metrics.hpp` helpers used in every `train_*` binary.

Open `results/voc/metrics_custom.csv` in a text editor. A detection run looks like:

```text
Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;mAP@0.5
1;142.351;140.118;12.041;13102;0.0021
2;118.204;121.553;11.887;13108;0.0048
```

Classification (`results/cifar10/metrics_custom.csv`) typically has `TrainLoss` / `TestLoss` and may have accuracy columns that `plot_metrics.py` **ignores** (any header containing `acc` is skipped during rename so `TrainAcc` cannot be mistaken for `TrainLoss`).

Tabular runs live under `results/tabular/`. Overfit and short VOC live under `results/overfit/` and `results/voc_short/`.

**How to interpret one row:**

- `Epoch` — 1-based, matches the training loop counter.
- `TrainLoss` / `TestLoss` — mean loss over the epoch as the binary defines it (YOLO grid loss vs cross-entropy). Custom and Torch must use the same reduction (`1/B` vs `sum`) or the curves will be scaled copies.
- `Time(s)` — host `std::chrono::steady_clock` around train + eval + logging for that epoch. Includes JPEG, H2D, `to_host` for mAP, checkpoint I/O if any. It is **not** a CUDA Event. First epoch is often slower (cuDNN find, `ensure`); quote epoch 2+ or report both.
- `VRAM_MiB` — `Profiler::get_vram_usage_mb()` once per epoch after a step. Custom YOLO should sit near **~13000** and be **flat**. A +200 MiB/epoch slope is a leak. Torch may sit lower; annotate with Chapter 3’s tradeoff sentence.
- `mAP@0.5` — VOC Custom currently writes this; Torch VOC CSV may omit it. Do not claim a detection-quality win without both columns.

If the file is comma-separated (Excel “save as CSV” on some locales), `plot_metrics.py` retries with `pd.read_csv` default comma when the semicolon parse yields a single column.

## 6.12 `plot_metrics.py` in thesis-production detail

```bash
python3 scripts/plot_metrics.py --results-root results
# optional:
python3 scripts/plot_metrics.py --results-root results --experiments voc,cifar10
```

Dependencies: matplotlib, pandas. The script forces `matplotlib.use("Agg")` so it runs headless inside Docker.

For each experiment directory it:

1. Loads `metrics_custom.csv` and `metrics_torch.csv` (missing Torch is a warning, not a crash—Custom-only plots are still written).
2. Normalises headers (`Loss` → `TrainLoss`, `VRAM_MiB`, `Time(s)`, `mAP@0.5`). Accuracy columns are skipped on purpose.
3. Writes into the experiment folder:
   - `plots/train_vs_test_loss.png`
   - `plots/map50.png` (if mAP exists)
   - `plots/epoch_duration.png`
   - `plots/vram.png`
4. Copies the same figures to `results/plots/<experiment>_*.png` as a gallery for the dissertation figure folder.

Style: serif fonts, 300 DPI, no top/right spines, dashed grid. Custom train is navy (`#1B4F72`); Torch train is maroon (`#922B21`). Keep these colours consistent across the thesis so a reader learns the legend once.

**Reading the four figures:**

| Figure | Healthy Custom vs Torch | Unhealthy |
| --- | --- | --- |
| Train vs test loss | Similar schedule, similar scale | Custom is a scalar multiple of Torch → missing `1/B`; diverges after epoch 5 → LR schedule mismatch (`scheduled_learning_rate` vs Torch `lr_schedule` in JSON) |
| Epoch duration | Custom within a few percent of Torch after epoch 1 | Custom 2× Torch → sync or copy regression; re-run `bench_micro_ops` |
| VRAM | Custom flat and high (~13 GiB YOLO); Torch may be lower | Custom rising → leak; Custom *lower* than Torch *and* slower → still allocating (warmup/`ensure` broken) |
| mAP | Both present, same eval protocol | Only Custom present → do not claim superiority |

## 6.13 Google Benchmark output: how to paste a table

```bash
ninja -C build bench_micro_ops
./build/benchmarks/bench_micro_ops \
  --benchmark_min_time=0.5s \
  --benchmark_counters_tabular=true \
  --benchmark_filter=FC
```

A typical tabular line (illustrative):

```text
-------------------------------------------------------------------------------------------------
Benchmark                       Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------
BM_Custom_FC_Forward         1.96 ms         1.96 ms          360 VRAM_MiB=13102 bytes_per_second=...
BM_Torch_FC_Forward          2.24 ms         2.24 ms          320 VRAM_MiB=11840
```

`Time` here is **GPU** time from `Profiler` via `SetIterationTime`, not `std::chrono` around the C++ for-loop. `VRAM_MiB` is sampled after the loop and is the **sum** of Custom + Torch contexts because `bench_micro_ops` links both in one process. Do not use that VRAM column as the YOLO training footprint; use the CSV `VRAM_MiB` from `train_voc_custom` instead. The micro-ops VRAM column is still useful to see that Custom caches are resident (large) versus a tiny op that forgot `ensure` (suspiciously small and slow).

`--benchmark_filter` is a regex. `FC`, `Conv`, `YOLOLoss`, `SGD` are useful slices. `--benchmark_min_time=0.5s` avoids 1-iteration turbo/idle noise. For the dissertation, pin clocks (`nvidia-smi -lgc` if the OEM allows), record driver version, and average several process launches.

Warmup (`kWarmup = 5`) runs before `benchmark::State`. Timing the first cuDNN `find` would punish Custom and credit Torch’s already-warm context. Never disable warmup for thesis tables.

## 6.14 Layer-by-layer VRAM in principle

The suite does not currently print a per-layer VRAM table automatically. To construct one for the thesis:

1. Run `train_voc_custom` with a one-batch smoke (or `overfit_voc_custom`).
2. After warmup, `Profiler::get_vram_usage_mb()` is the full process.
3. Optionally add temporary logs after each `FusedCBR2d` / `FullyConnected` constructor and after first forward; the *delta* is that layer’s caches + workspace.
4. Nsight Compute / `cudaMemGetInfo` around a single `layer->forward` is the same idea.

The important plot is still the **epoch-flat** CSV curve. Per-layer deltas explain *why* 13 GiB (804 MiB FC weights, large stem activations, cuDNN workspaces) rather than to hunt a leak that the flat curve already disproves.

## 6.15 Interactive menu

`scripts/menu.sh` groups Custom/Torch pairs (historically items 0–28 after the symmetry refactor). It:

- sources `cuda_env.sh` so the cached CUDA arch matches the live GPU;
- reconfigures CMake if they diverge;
- `ninja -C build <target>` then runs the binary from the build tree so `DEEPLEARN_SOURCE_DIR` finds `config/experiments.json`;
- **Run all** skips missing Torch binaries;
- **Sanity Check** exports `EXPERIMENTS_JSON=config/sanity.json` (2 epochs, `dataloader_workers: 0`).

Sanity is a rebuild smoke test. Quote only full `experiments.json` runs, or micro-ops, in dissertation tables.

## 6.16 Suggested filled thesis table (template)

Record in the same session: driver, `nvidia-smi` clocks, batch 16, `CMAKE_CUDA_ARCHITECTURES` (`120-real` or whatever `dev.sh` printed), NGC tag `26.03-py3` if Docker.

| Operator | Custom ms | Torch ms | Notes |
| --- | --- | --- | --- |
| FC forward | | | `matmul_into`, TF32 |
| FC backward | | | logical `OP_T`, no 804 MiB copy |
| Conv 3×3 @ 112, 64→192 | | | cuDNN find cached |
| YOLOLoss | | | device `yolo_mean_loss_kernel` |
| YOLO train step (micro) | ~ | ~ | excludes dataloader |
| YOLO epoch (CSV, epoch ≥ 2) | ~77.8 ms class of result | ~76.4 ms class of result | includes I/O; replace with *your* CSV |
| VRAM after warmup | ~13000 MiB | (Torch CSV) | Chapter 3 tradeoff |

Replace the illustrative 77.8 / 76.4 with the numbers from **your** pinned-clock run. Those figures were measured on a development RTX-class GPU after the fused-SGD / GPU-loss-reduce round; they are a class of result, not a universal constant.

## 6.17 What not to measure (normative list)

- First iteration after process start (cuDNN find, `cudaMalloc`).
- Debug + ASan builds (host sanitizers destroy GPU timing).
- Mixed Custom/Torch VRAM in `bench_micro_ops` as if it were YOLO train VRAM.
- `extern/pytorch` submodule rebuilds as Custom performance work.
- `config/sanity.json` epoch times.
- Clock-boosted single-run outliers; report mean ± range or pin clocks.

The benchmark story of this thesis is: **same GPU, same batch, Custom kernels and caches versus LibTorch, allocator traffic removed, I/O overlapped, numbers in CSV and micro-ops, VRAM reported honestly.**
