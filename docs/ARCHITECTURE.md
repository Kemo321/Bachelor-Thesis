# Architecture

DeepLearnLib is a **generic** GPU tensor library. Networks are graphs of `Layer` objects; YOLO and SimpleCNN are applications that sit on top of that graph. This document describes the performance contract of the core: keep compute on the device, reuse memory, and overlap host decode with GPU work.

## Core vs applications

| Piece | Location | Linked into |
| --- | --- | --- |
| `dl::Tensor`, `Layer`, Conv/Pool/BN/FC, losses | `include/DeepLearnLib/`, `src/` | `DeepLearnLib` |
| YOLOv1, SimpleCNN | `benchmarks/models/` | `DeepLearnModels` (apps + tests only) |

The core stays small because topologies are not compiled into the library. A new detector or classifier is a new model file, not a core change. See [ADDING_LAYERS.md](ADDING_LAYERS.md) for extending the layer set.

`Network` is a thin ordered stack plus checkpoint I/O. Its `fit()` helper uses `YOLOLoss` for detection grids; classification pipelines run their own loop with `CrossEntropyLoss`.

## `dl::Tensor` memory

Storage is a `std::shared_ptr<float>` with `CudaDeleter` (`cudaFree`). Views (`view`, `as_view`) share that pointer; they do not copy device memory. Host traffic is explicit:

- `from_host` — H2D (dataloader upload)
- `to_host` — D2H (logging, mAP, CSV). This **synchronises** the stream.

Training kernels never bounce activations to the CPU. Mixed precision can store FP16 (`__half`) on device while `to_host` still returns FP32.

### Why higher VRAM is accepted

LibTorch is fast partly because it **caches** workspaces and keeps the allocator off the hot path. DeepLearnLib does the same, explicitly:

- Each parameterised layer holds `std::optional<dl::Tensor>` caches for output, input, and `dL/dx`.
- `Tensor::ensure(slot, shape, device, dtype)` reallocates **only** when shape, device, or dtype change.
- With a fixed batch size the first step pays `cudaMalloc`; later steps reuse the same buffers.

That increases the process's VRAM footprint (activations live for backward; GEMM workspaces stay reserved) and removes allocator stalls and extra D2D copies. The thesis comparison is wall-clock per epoch against LibTorch, not minimum-memory inference.

## Zero-allocation GEMM: `matmul_into`

`FullyConnected` is a pair of GEMMs. A naive `C = A.matmul(B)` would `cudaMalloc` `C` every call.

```text
forward:  Y = X @ W          ->  X.matmul_into(W, Y_cache)
dW:       dW = X^T @ dY      ->  X.matmul_into(dY, dW, /*transA=*/true, false, momentum)
dX:       dX = dY @ W^T      ->  dY.matmul_into(W, dX_cache, false, /*transB=*/true)
```

Transpose flags map to `CUBLAS_OP_T`. There is **no** physical `transpose()` allocation; cuBLAS reads the same row-major buffer as a swapped column-major view.

`beta` lets the weight-gradient GEMM accumulate momentum into the existing `dW` buffer (`Y = A B + beta Y`).

A 64 MiB cuBLAS workspace is attached to the process handle so algorithm picking does not allocate per call.

## In-place updates

Elementwise helpers mutate storage in place (`add_`, `mul_`, `add_scaled_`, `clamp_`, `sgd_update_`). SGD is fused:

```text
w -= lr * clip(grad + decay * w, [-clip, clip])
```

Weight decay is **not** applied in `backward()` (that would need extra buffers). `Layer::step()` calls `sgd_update_` on weights and biases.

## Kernel fusion

| Fusion | Where | Effect |
| --- | --- | --- |
| Conv + bias (+ identity act) | `FusedCBR2d` / `cudnnConvolutionBiasActivationForward` | One cuDNN launch instead of conv then bias |
| BN affine + LeakyReLU | custom CUDA kernel in `FusedCBR2d` | No extra global-memory round trip |
| SGD + decay + clip | `Tensor::sgd_update_` | One kernel over the parameter buffer |
| YOLO cell loss reduce | `YOLOLoss` device reduce | No `thrust::reduce` host sync per batch |

Dropout, Softmax, and LeakyReLU write into layer-owned workspaces (`ensure`), not fresh tensors.

## Prefetching thread pool (`CustomDataLoader`)

JPEG decode and YOLO target packing are CPU-bound. The loader:

1. Uses a **bounded thread pool** to decode the current mini-batch on the host.
2. Starts a `std::future` for the **next** host batch (`launch_prefetch`) while the GPU trains on the current `Batch`.
3. Uploads with `Tensor::from_host` onto a caller-provided CUDA stream.

`ClassificationLoader` uses the same pattern for ImageFolder-style CIFAR data.

Training loops can double-buffer further: two `UniqueCudaStream`s, overlap H2D of batch *n+1* with compute on batch *n* (`train_voc_custom`, `train_cifar_custom`).

The invariant is: **CPU decode never sits on the CUDA default stream**, and **GPU compute does not wait on JPEG I/O** except at the first batch of an epoch.

## Streams and cuDNN

`dl::StreamGuard` rebinds `current_stream()` for a lexical scope. `bind_cudnn_stream` points the process cuDNN handle at the same stream so conv/pool/BN enqueue with GEMM and custom kernels. Layers should not call `cudaDeviceSynchronize` on the training path.

## Related code

- Tensor API: `include/DeepLearnLib/Tensor.hpp`
- Layer contract: `include/DeepLearnLib/Layer.hpp`
- Dataloader: `include/DeepLearnLib/dataset.hpp`, `ClassificationLoader.hpp`
- Fused backbone block: `include/DeepLearnLib/FusedCBR2d.hpp`
