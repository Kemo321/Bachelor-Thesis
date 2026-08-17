# Chapter 5 — Asynchronous dataloading

## 5.1 The CPU problem

YOLOv1 training at 448×448 is not only a GPU problem. Each training sample requires:

1. reading a JPEG from disk;
2. OpenCV decode to BGR;
3. resize to 448×448;
4. optional affine scale/translation and HSV jitter (`is_train == true`);
5. RGB conversion, `[0,1]` scaling, HWC→CHW packing;
6. parsing labels (VOC XML via pugixml, or YOLO txt) into a 7×7×(10+C) target grid.

On a single thread this work exceeds the GPU's ~20 ms Custom forward for a batch of 16. The GPU then idles in `from_host` waiting for the CPU. LibTorch hides the same problem behind `DataLoader(num_workers=N)`. DeepLearnLib must solve it without `torch::data`.

There are two overlapping mechanisms, and they are easy to confuse:

| Mechanism | Where | Overlaps |
| --- | --- | --- |
| Bounded thread pool `dl::parallel_for` | inside one host batch | JPEG decode of sample *i* with sample *j* **in the same batch** |
| `std::future` prefetch | `launch_prefetch` / `get_batch` | decode of batch **N+1** with GPU compute on batch **N** |
| Double-buffered CUDA streams | `train_voc_custom`, `train_cifar_custom` | H2D of batch **N+1** with GEMM/conv on batch **N** |

All three exist. Omitting any one of them reintroduces a bubble.

## 5.2 Bounded thread pool: `parallel_worker_count` and `parallel_for`

`include/DeepLearnLib/ParallelFor.hpp`:

```cpp
inline auto parallel_worker_count(int work_items) -> int
{
    if (work_items <= 1) return 1;
    unsigned hardware = std::thread::hardware_concurrency();
    int cap = (hardware == 0) ? 8 : (int)hardware;
    cap = min(cap, 16);
    return min(work_items, cap);
}
```

**Why a cap of 16?** CIFAR batch 64 with one `std::async` per image creates 64 OS threads, each calling OpenCV. The kernel scheduler thrashes; decode *slows down*. Sixteen workers striding through 64 indices (`index += workers`) keep the CPU saturated without oversubscription. Tests lock this contract: `parallel_worker_count(1)==1` and `parallel_worker_count(128)<=16`.

`parallel_for(count, fn)` launches `workers` async tasks. Each worker runs `fn(index)` for `index = worker, worker+workers, ...`. Exceptions are captured with `std::exception_ptr` and rethrown on the joining thread so a corrupt JPEG fails the batch instead of terminating the process in a worker.

`CustomDataLoader::decode_job` fills `host.images` and `host.targets` then:

```cpp
dl::parallel_for(host.n, [&](int batch_idx) {
    mt19937 local_rng(rng_seeds[batch_idx]);
    load_sample(sample_indices[batch_idx], sample_image, sample_target, local_rng);
    copy into host buffers at batch_idx * elems
});
```

Each sample has a **pre-drawn RNG seed** (`take_job` packs `rng_seeds`) so that parallel jitter is deterministic per index and does not share a `mt19937` across threads (which would be a data race).

`ClassificationLoader` uses the same `parallel_for` over a batch of ImageFolder JPEGs (`parallel_worker_count(batch_size_)`).

## 5.3 Prefetch with `std::future`

Both loaders hold `std::future<HostBatch> prefetch_`.

**`launch_prefetch`:** if there is a remaining job (`take_job` / `take_indices`), start

```cpp
prefetch_ = std::async(std::launch::async, [indices, seeds] {
    return decode_job(indices, seeds);
});
```

`std::launch::async` is mandatory. `deferred` would decode on the GPU thread inside `.get()`, destroying the overlap.

**`get_batch(stream)`:**

```text
if (!prefetch_ valid) launch_prefetch()     // first call of an epoch
host = prefetch_.get()                      // wait for CPU batch N
prefetch_ = {}
launch_prefetch()                           // start CPU batch N+1 NOW
return upload_host_batch(host, stream)      // H2D on the provided stream
```

Timeline for a naive single-stream loop:

```text
CPU:  [decode N][decode N+1][decode N+2]
GPU:            [H2D N][fwd+bwd N][H2D N+1][fwd+bwd N+1]
```

After prefetch, decode N+1 overlaps fwd+bwd N:

```text
CPU:  [decode 0][    decode 1    ][    decode 2    ]
GPU:            [H2D0][compute 0 ][H2D1][compute 1 ]
```

If decode(N+1) is slower than compute(N), the GPU still waits in `.get()`. The thread pool exists to make decode(N) shorter than compute(N) on the thesis hardware. If it is not, the correct next step is more decode workers or faster JPEG (libjpeg-turbo), not more CUDA Graphs.

**`reset()`** joins any outstanding future (`join_prefetch`) and reshuffles `order_` for training. **`join_prefetch`** swallows wait exceptions then resets the future so the destructor never blocks on a detached task after a failed epoch.

## 5.4 Upload: `from_host` on a chosen stream

`upload_host_batch` constructs:

```text
images  [N, 3, 448, 448]   from_host(..., stream, compute_dtype())
targets [N, 7, 7, 10+C]    from_host(..., stream, compute_dtype())
```

Passing `stream` means the DMA is enqueued on that stream, not the default stream (which would serialise with everything else). `compute_dtype()` selects FP32 or FP16 storage according to `dl::configure_precision`.

Pinned host staging (`cudaMallocHost` in `Tensor.cpp`) allows `cudaMemcpyAsync`. Without pinning, the driver may silently stage through an extra bounce buffer and lose overlap.

## 5.5 Double-buffered CUDA streams in the training loop

Prefetch overlaps **CPU decode** with **GPU compute**. It does **not** overlap **H2D** with compute if both use the same stream: CUDA streams are FIFO. `train_voc_custom.cpp` therefore keeps two `UniqueCudaStream` objects:

```cpp
dl::UniqueCudaStream copy_streams[2];
optional<Batch> batches[2];
bool has_batch[2] {false, false};

// prime slot 0
batches[0] = train_loader.get_batch(copy_streams[0].get());
has_batch[0] = true;

int slot = 0;
while (has_batch[slot]) {
    int next = 1 - slot;
    cudaStream_t compute_stream = copy_streams[slot].get();
    CHECK_CUDA(cudaStreamSynchronize(compute_stream));  // H2D of this slot done

    StreamGuard guard(compute_stream);
    pred = model.forward(batches[slot]->images, compute_stream);
    loss = YOLOLoss::loss(..., compute_stream);
    // backward + step on compute_stream

    if (train_loader.has_next()) {
        cudaStreamSynchronize(copy_streams[next].get());
        batches[next] = train_loader.get_batch(copy_streams[next].get());
        has_batch[next] = true;
    } else {
        has_batch[next] = false;
    }
    slot = next;
}
```

Interpretation:

- Slot *s* owns a stream. `get_batch(stream_s)` enqueues H2D on `stream_s`.
- Before using `batches[s]`, the loop synchronises **that** stream, not the whole device.
- While the GPU computes on slot *s*, the CPU thread can `get_batch` on slot `1-s`, which starts decode (prefetch inside the loader) and H2D on the other stream.
- `UniqueCudaStream` is non-blocking (`cudaStreamNonBlocking`) so these streams do not wait for the legacy default stream.

`train_cifar_custom.cpp` uses the same idea via `for_each_prefetched_batch`, which encapsulates the two-stream ping-pong for `ClassificationLoader`.

```text
Stream 0:  H2D batch0 | compute batch0 | H2D batch2 | compute batch2
Stream 1:              H2D batch1 | compute batch1 | H2D batch3 | ...
CPU:       decode0     decode1         decode2         decode3
```

The `StreamGuard` plus `bind_cudnn_stream` ensures cuDNN/cuBLAS enqueue on `compute_stream` rather than stream 0.

## 5.6 VOC-specific CPU work

`load_sample` (CustomDataLoader) still does geometry jitter and HSV on the CPU because those ops are cheap compared to JPEG decode and because they produce the CHW float buffer that `from_host` consumes. Moving jitter to a GPU kernel would save little until decode is no longer the limiter.

XML conversion (`convert_voc_to_yolo`) is a one-time filesystem side effect: Pascal VOC annotations become YOLO txt next to JPEGs so the hot path does not parse XML every epoch.

## 5.7 Interaction with logging

`YOLOLoss::loss(...).to_host(compute_stream)` in the VOC loop **synchronises `compute_stream`** to print the batch loss. That is a conscious leak of overlap for observability. Production-like timing runs should log every N batches (the loop already logs batch 1 and every 50). Micro-benchmarks (`bench_voc_*`) do not print per batch.

## 5.8 Failure modes

| Symptom | Likely cause |
| --- | --- |
| GPU util low, CPU 100% | `parallel_for` cap too small or disk is cold |
| GPU util low, CPU idle | missing prefetch (`get_batch` decodes on the GPU thread) or all work on default stream |
| H2D not overlapped | `from_host` without stream; or sync on default stream; or unpinned host memory |
| Data race / flickering augment | shared RNG instead of per-sample seeds |
| Hang on shutdown | `prefetch_` not joined in destructor/`reset` |

## 5.10 `CustomDataLoader` lifetime of a batch

The VOC / BCCD / synthetic detection loader is `CustomDataLoader` in `src/dataset.cpp` (OpenCV-gated). Its public training API is small:

```text
reset()           // join prefetch, reshuffle order_ if train
has_next()        // remaining jobs in this epoch
get_batch(stream) // HostBatch → GPU Batch on `stream`
```

Internally:

1. **Index list.** `order_` is a permutation of sample indices. Training reshuffles each `reset()`. Eval is sequential.
2. **Job.** `take_job()` pops the next `batch_size` indices (or fewer at the tail) and draws a `uint32_t` RNG seed per index from the loader’s `mt19937`.
3. **Prefetch future.** `launch_prefetch` starts `std::async(std::launch::async, decode_job)`.
4. **`decode_job`.** Allocates host `images` / `targets` float buffers, then `dl::parallel_for(n, [&](batch_idx){ load_sample(...); memcpy into slot; })`.
5. **`load_sample`.** `cv::imread` JPEG, resize to 448×448, optional affine + HSV jitter if `is_train`, BGR→RGB, `/255`, HWC→CHW, read YOLO txt labels into a 7×7×(10+C) grid (objectness, box, one-hot class).
6. **`upload_host_batch`.** `Tensor::from_host` on the provided CUDA stream, dtype `compute_dtype()`.

VOC XML is **not** parsed on the hot path. `convert_voc_to_yolo` is a one-time filesystem conversion: Pascal annotations become YOLO txt beside JPEGs. Repeating pugixml every epoch would add CPU work that does not overlap well (XML is not parallelised per box in the same way JPEG is).

## 5.11 Why JPEG is the bottleneck and why a cap of 16 workers

OpenCV’s `imread` of a VOC JPEG (often ~500×375, sometimes larger) plus `cv::resize` to 448×448 dominates `load_sample`. Colour jitter is cheap. Grid packing is cheap. The GPU Custom forward for batch 16 is ~20 ms in micro-ops; sixteen serial JPEGs on one core can exceed that.

A naive `std::async` per image at CIFAR batch 64 creates 64 OS threads, each in libpng/libjpeg and OpenCV. The kernel scheduler thrashes: context-switch cost exceeds decode work; last-level cache is blown; decode *slows down*. `parallel_worker_count` therefore:

```text
cap = min(hardware_concurrency or 8, 16)
return min(work_items, cap)
```

Workers stride: worker `w` handles indices `w, w+workers, w+2*workers, …`. Tests lock `parallel_worker_count(1)==1` and `parallel_worker_count(128)<=16`.

Exceptions: each worker’s `fn(index)` is inside `std::async`. `parallel_for` joins all futures, captures the first `exception_ptr`, and rethrows on the calling thread. A corrupt JPEG fails the batch instead of `std::terminate` in a worker.

RNG: sharing one `mt19937` across workers is a data race. `take_job` pre-draws seeds; each `load_sample` constructs a local `mt19937(seed)`. Augmentation is deterministic per (epoch shuffle, index) pair.

`ClassificationLoader` uses the same `parallel_for` over ImageFolder JPEGs (`parallel_worker_count(batch_size_)`). CIFAR’s 32×32 decode is faster per image; the cap still matters at batch 64.

## 5.12 Prefetch state machine

Both loaders hold `std::future<HostBatch> prefetch_`.

```text
                    ┌──────────────┐
           reset    │  idle        │
         ──────────►│ prefetch_    │
                    │  invalid     │
                    └──────┬───────┘
                           │ first get_batch
                           ▼
                    ┌──────────────┐
                    │ launch N     │
                    │ wait N       │
                    │ launch N+1   │  ◄── loop
                    │ upload N     │
                    └──────────────┘
```

`std::launch::async` is mandatory. `std::launch::deferred` (or default `async|deferred` on some libraries) would decode inside `.get()` on the GPU thread and destroy overlap.

`join_prefetch` in `reset` and the destructor swallows wait exceptions then resets the future so destruction never blocks on a detached task after a failed epoch. A hang on shutdown is almost always a missing join.

If `decode(N+1)` is slower than `compute(N)`, the GPU waits in `.get()`. The thread pool exists to make decode shorter than compute on the thesis hardware. If it is not, the next step is faster JPEG (libjpeg-turbo) or more disks, not CUDA Graphs.

## 5.13 Double-buffer streams: why two is the right number

Prefetch overlaps **CPU decode** with **GPU compute**. It does **not** overlap **H2D** with compute if both use the same stream: CUDA streams are FIFO. `train_voc_custom.cpp` therefore keeps two `UniqueCudaStream` objects (`cudaStreamNonBlocking`):

```text
Slot 0 stream:  H2D0 | compute0 | H2D2 | compute2 | …
Slot 1 stream:        H2D1 | compute1 | H2D3 | compute1 …
CPU threads:    dec0  dec1      dec2      dec3
```

Before compute on slot `s`, the loop `cudaStreamSynchronize(copy_streams[s])` — **that** stream only, not the device. While compute runs on `s`, the CPU may `get_batch` on `1-s`, which starts decode (via prefetch) and enqueues H2D on the other stream.

`StreamGuard` + `bind_cudnn_stream` + `cublasSetStream` (inside `launch_rowmajor_gemm`) bind compute to `copy_streams[s]`. Using the default stream would wait for both copy streams (legacy null-stream semantics) and collapse the overlap.

`train_cifar_custom.cpp` uses `for_each_prefetched_batch`, which encapsulates the same ping-pong for `ClassificationLoader`.

Three streams would allow a further H2D/compute/compute split only if the GPU had a separate copy engine *and* the CPU could decode fast enough to fill it. On the thesis hardware two streams saturated the overlap; a third added complexity without measured win.

## 5.14 Pinning and `from_host`

`upload_host_batch` constructs:

```text
images  [N, 3, 448, 448]  from_host(..., stream, compute_dtype())
targets [N, 7, 7, 10+C]   from_host(..., stream, compute_dtype())
```

Pinned host staging (`cudaMallocHost`) allows `cudaMemcpyAsync`. Unpinned pageable memory causes the driver to stage through a bounce buffer; the copy then typically synchronises, and Chapter 5’s overlap diagram lies. Destruction of pinned buffers synchronises the associated stream (`PinnedHostDeleter`, Chapter 3).

## 5.15 Logging versus overlap

`YOLOLoss::loss(...).to_host(compute_stream)` synchronises `compute_stream` to print the batch loss. That is a conscious leak of overlap for observability. The VOC loop logs batch 1 and every 50. Micro-benchmarks (`bench_voc_*`) do not print per batch. Thesis epoch times in CSV use host `steady_clock` around the whole epoch (Chapter 6), which *includes* those syncs; that is the number a user experiences.

## 5.16 Failure mode table (expanded)

| Symptom | Likely cause | Check |
| --- | --- | --- |
| GPU util low, CPU 100% | `parallel_for` cap too small; cold disk | `nvidia-smi dmon`; `iostat` |
| GPU util low, CPU idle | missing prefetch; work on default stream | Nsight: `DataLoader_GetBatch` before every forward with GPU idle |
| H2D not overlapped | `from_host` without stream; unpinned host | Nsight memcpy vs GEMM tracks |
| Flickering augment / races | shared RNG | per-sample seeds in `take_job` |
| Hang on shutdown | `prefetch_` not joined | `join_prefetch` in `reset`/dtor |
| `bus error` in container | `/dev/shm` too small | compose `shm_size: 32gb` |
| Custom slower only on first batch of epoch | expected: `ensure` + cuDNN find | exclude from micro-ops via warmup |

## 5.17 Fair comparison with LibTorch workers

Torch pipelines set `dataloader_workers` in `experiments.json` (4 in full runs, 0 in `sanity.json`). Custom’s `std::async` pool is the analogue of `num_workers>0`. Custom’s two CUDA streams are the analogue of `non_blocking=True` plus a side stream. The thesis comparison is fair only when both sides overlap I/O. Sanity configs disable workers to make smoke tests deterministic and lighter on CI-like environments. **Do not quote sanity epoch times as performance results.**
