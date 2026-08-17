# Chapter 4 — Kernel fusion and compute

## 4.1 Compute philosophy

Once `ensure` has removed allocator traffic (Chapter 3), remaining epoch time is **kernel launches and memory bandwidth**. Two classes of waste dominated early Custom profiles:

1. **Launch and synchronisation tax** of Thrust algorithms that look like one-liners (`thrust::transform`, `thrust::reduce`) but insert implicit device-wide waits when they return a host `float`.
2. **Global-memory round trips** between mathematically adjacent maps: convolution then bias then BatchNorm affine then LeakyReLU as four kernels, each reading and writing a `[N,C,H,W]` activation.

DeepLearnLib's response is not “write everything as one mega-kernel.” Convolution is still cuDNN; GEMM is still cuBLAS. The response is: **fuse what is elementwise and adjacent**, and **never reduce on the host** in the loss.

## 4.2 Thrust: what remains and what was purged

Thrust is still linked (it ships with the CUDA toolkit). It is acceptable for:

- one-off fills of a buffer during initialisation (`thrust::fill` in `FusedCBR2d` setup);
- allocating binary operators (`Tensor::operator+`) used by tests.

It is **not** acceptable on the training hot path for:

- SGD: replaced by `sgd_update_f32_kernel`;
- in-place add/scale: `add_scaled_inplace_f32_kernel`, `add_` kernels;
- YOLO loss mean: replaced by `yolo_mean_loss_kernel` (shared-memory reduction, writes a device scalar).

The historical bug is worth stating in thesis language. `thrust::reduce` on a `device_ptr` returns a host `T`. That return is a synchronisation point: the CPU cannot continue until the entire grid finishes and the value is copied. Calling it once per batch inside `YOLOLoss::loss` added a full pipeline stall *in addition to* whatever `to_host` the logger already needed. After the change, `loss()` returns a 1-element GPU tensor; the training loop may `to_host` it when printing, which is one sync per printed batch, not one sync plus a hidden Thrust sync.

Elementwise Thrust `transform` without a reduce is often asynchronous, but it still:

- instantiates a heavy template backend;
- cannot fuse decay + clip + add;
- tends to allocate if used in the allocating `operator+` style.

Bare-metal `__global__` kernels in `Tensor.cpp` have a trivial launch configuration (`kInplaceThreads`, typically 256), take raw pointers, and are instantiated twice (FP32 / FP16) with explicit `__half2float` round trips where needed.

## 4.3 In-place SGD kernel

```cuda
__global__ void sgd_update_f32_kernel(float* weights, const float* grad,
                                      float lr, float decay, float clip, int count)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= count) return;
    float update = grad[index] + (decay * weights[index]);
    if (clip > 0.0F)
        update = update < -clip ? -clip : (update > clip ? clip : update);
    weights[index] -= lr * update;
}
```

Properties:

- **One load of `w` and `g`, one store of `w`.** A three-kernel version (decay into temp, clip into temp, axpy) would be three times the bandwidth.
- **`clip <= 0` disables clipping** without a separate code path on the host other than the branch inside the kernel (uniform across the warp for a given launch).
- **Decay lives here**, not in `backward()`. `FullyConnected::step` passes `kWeightDecay = 0.0005F` and `parameter_clip_bound()`.
- Mixed precision: the FP16 kernel computes in float and writes `__float2half`.

`Layer::scaled_learning_rate()` divides `learning_rate` by `dl::loss_scale()` so that AMP does not require a second kernel.

## 4.4 `FusedCBR2d`: convolution, bias, BatchNorm, LeakyReLU

YOLOv1's backbone is not a stack of four independently launched primitives. `FusedCBR2d` is the unit of reuse in `benchmarks/models/YOLO.cpp` (`add_block` → `FusedCBR2d`).

### 4.4.1 cuDNN convolution + bias

`Conv2d` (embedded as `conv_`) still owns filters and the cuDNN convolution descriptor. Forward uses `cudnnConvolutionBiasActivationForward` with **IDENTITY** activation. IDENTITY is mandatory: BatchNorm must see a *linear* pre-activation. Fusing ReLU into cuDNN here would make the BN statistics wrong.

cuDNN picks algorithms on first use (`cudnnFindConvolutionForwardAlgorithm` / the workspace `CudaWorkspace::ensure` in `Conv2d.cpp`). That workspace is retained. TF32 is enabled at the global cuDNN benchmark context from Torch baselines and via math modes on the Custom side.

### 4.4.2 Custom BN affine + LeakyReLU kernel

After convolution, `apply_bn_leaky_into` runs a single elementwise kernel over NCHW that, per channel `c`:

```text
xhat = (x - mean[c]) * inv_std[c]
y    = gamma[c] * xhat + beta[c]
out  = y > 0 ? y : leaky_slope * y
```

in **one** load of `x` and one store of `out` (plus broadcast loads of the 1D BN vectors, which hit cache). The alternative—`BatchNorm2d::forward` then `LeakyReLU::forward`—writes the BN output to global memory and immediately rereads it. At YOLO spatial sizes that extra round trip is measurable.

Training versus eval: running mean/variance updates remain in the BN portion (cuDNN `cudnnBatchNormalizationForwardTraining` / inference, or the custom statistics path associated with `save_mean_` / `save_inv_var_`). The fused elementwise kernel consumes those saved moments. Backward reconstructs the BN backward then the conv backward; caches `bn_input_cache_`, `fused_output_cache_`, `grad_bn_cache_`, `grad_conv_cache_` are `ensure`d (Chapter 3).

### 4.4.3 Why not fuse the convolution into the same CUDA file

A handwritten NCHW GEMM-unrolled conv would lose cuDNN's Winograd / implicit GEMM selection, which *is* LibTorch's backend. The thesis fuses **around** cuDNN, not instead of it. That is the correct division of labour: NVIDIA owns convolution performance; the student owns the glue that LibTorch also glues (but hides).

## 4.5 Other fused or cache-heavy layers

- **Dropout** generates a mask into `mask_` (`ensure`) and scales. Eval skips the mask. The mask lives until backward.
- **Softmax** writes probabilities into `output_cache_` and uses it for the Jacobian in backward.
- **LeakyReLU** (standalone, used in YOLO head and SimpleCNN) is a one-kernel map with cached input for the backward mask `x > 0`.
- **MaxPool2d** uses cuDNN pooling; indices/workspace follow cuDNN's training mode.

## 4.6 `YOLOLoss`: single-pass cell kernel plus device reduce

### 4.6.1 Layout

Predictions and targets are `[B, 7, 7, 10+C]` or flattened `[B, 7*7*(10+C)]`. `as_yolo_grid` views either as the 4D grid. Each cell holds two boxes of 5 numbers (x, y, w, h, conf) plus `C` class scores (`CLASS_OFFSET = 10`).

### 4.6.2 Forward kernel

`yolo_loss_forward_kernel` assigns **one thread per cell** (`cell_count = B * 49`). That thread:

- computes IoU of both predicted boxes against the target box (`yolo_iou` device function / `yolo_iou_kernel` for the standalone API);
- selects the responsible box;
- accumulates localisation (xy, sqrt-wh), objectness, no-object, and classification terms with the standard YOLOv1 λ constants;
- writes a **scalar cell loss** into `cell_loss[cell]`.

No host branch on objectness. The object mask is data.

### 4.6.3 Reduction kernel

```cuda
__global__ void yolo_mean_loss_kernel(const float* cell_loss, float* mean_loss,
                                      int cell_count, float inv_batch)
{
    __shared__ float shared_sum[kThreads];
    float partial = 0;
    for (int i = threadIdx.x; i < cell_count; i += blockDim.x)
        partial += cell_loss[i];
    shared_sum[threadIdx.x] = partial;
    __syncthreads();
    // tree reduce in shared memory
    if (threadIdx.x == 0)
        mean_loss[0] = shared_sum[0] * inv_batch;
}
```

Launched as `<<<1, kThreads>>>`. For `B=16`, `cell_count=784`, one block is enough. The output is a GPU tensor of shape `[1]`. **There is no `thrust::reduce`.** The logger's `to_host` is the first host wait if the loop prints.

Workspace is process-static:

```cpp
struct YoloWorkspace {
    optional<Tensor> cell_loss, grad, scalar;
};
```

`ensure` on those three buffers means YOLO loss allocates at most once per unique `(B, C)` pair.

### 4.6.4 Backward kernel

`yolo_loss_backward_kernel` writes `dL/dpred` with the same responsible-box logic, scaled by `inv_batch`. Mixed precision then multiplies by `loss_scale()` if AMP is on. The gradient tensor is `ensure`d and returned as a view.

### 4.6.5 IoU

`calculate_iou` launches `yolo_iou_kernel` for `[N,4]` boxes. Training uses the inlined IoU inside the cell kernel to avoid an extra global buffer of IoUs.

## 4.7 NVTX ranges

Hot functions wrap `dl::NvtxRange("FullyConnected_Forward")` and similar. Nsight Systems then shows Custom FC versus Torch FC on the same timeline. This is how the thesis attributes remaining milliseconds after fusion: GEMM versus conv versus loss versus H2D.

## 4.8 What “matching LibTorch compute speed” means numerically

After fused SGD, GPU loss reduce, `matmul_into`, TF32, and cache reuse, measured Custom versus Torch YOLO training at batch 16 sat at approximately **77.8 ms versus 76.4 ms** per epoch on the development RTX-class GPU, with Custom forward sometimes slightly faster than Torch (20.5 vs 23.5 ms in one micro pass) and FC backward comparable (1.96 vs 2.24 ms). Those numbers will move with driver and clocks; Chapter 6 describes how to reproduce them. The architectural point is that the remaining gap is **not** an allocator gap and **not** a Thrust-reduce gap. Further hypothetical wins (NHWC, CUDA Graphs) were estimated at ~5–10% and were left unimplemented so the thesis can defend a finished, auditable stack rather than an unbounded optimisation backlog.

## 4.10 Elementwise kernel catalogue in `Tensor.cpp`

The Thrust purge on the hot path is not a slogan; it is a list of `__global__` functions that replaced `thrust::transform` / `thrust::for_each` in the optimiser and in-place arithmetic. Each kernel is instantiated for `float` and `__half`. FP16 kernels convert to float for the arithmetic and write `__float2half`, because LeakyReLU-style branches and clips are not faster in native half on all thesis GPUs, and because numerical behaviour then matches the FP32 path besides storage.

Launch hygiene is uniform:

```cuda
kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(...);
CHECK_CUDA(cudaGetLastError());
```

`conversion_launch` is `ceil(count / kInplaceThreads)`. Shared memory is 0 bytes: these maps are purely register + global. Occupancy is limited by memory bandwidth, not by registers. Fusing decay, clip, and add in `sgd_update_*` is therefore a **bandwidth** fusion: one load of `w`, one load of `g`, one store of `w`, versus three round trips.

`add_row_*` implements bias add without a GEMM: `dst[i] += bias[i % features]`. `add_sum_rows_*` is the backward: each output column sums over the batch, with `beta` so that momentum on bias gradients matches the weight GEMM’s `beta`.

Allocating `operator+` remains Thrust. Grep `thrust::` in `src/Tensor.cpp` should show it on that path and on initialisation fills, not inside `sgd_update_` or `add_`.

## 4.11 `FusedCBR2d` dataflow, kernel by kernel

YOLOv1’s backbone is a stack of `FusedCBR2d` blocks. The name means Convolution + Bias + (BatchNorm affine) + (leaky) ReLU. The implementation is *not* one kernel. It is a deliberate sandwich:

```text
                    ┌─────────────────────────────────────┐
  x NCHW  ────────► │ cuDNN conv + bias (IDENTITY act)    │
                    │ cudnnConvolutionBiasActivationForward│
                    └──────────────────┬──────────────────┘
                                       │ linear pre-activation
                                       ▼
                    ┌─────────────────────────────────────┐
                    │ spatial_moments_kernel  (train)     │
                    │   one block per channel             │
                    │   shared-memory reduce sum, sum_sq  │
                    └──────────────────┬──────────────────┘
                                       ▼
                    ┌─────────────────────────────────────┐
                    │ finalize_bn_stats_kernel            │
                    │   inv_std = rsqrt(var + eps)        │
                    │   EMA into running_mean/var         │
                    └──────────────────┬──────────────────┘
                                       ▼
                    ┌─────────────────────────────────────┐
                    │ fused_bn_leaky_kernel               │
                    │   y = leaky(γ * xhat + β)           │
                    │   one load of x, one store of y     │
                    └─────────────────────────────────────┘
```

### 4.11.1 Why IDENTITY in cuDNN

`cudnnConvolutionBiasActivationForward` *can* fuse ReLU. FusedCBR2d passes **IDENTITY**. BatchNorm’s mean and variance must be computed on the *linear* convolution output. If ReLU were fused into cuDNN, the moments would be moments of a rectified tensor, which is a different (wrong) estimator. LibTorch’s `Conv2d` + `BatchNorm2d` + `LeakyReLU` module stack has the same constraint; PyTorch’s `cudnn.benchmark` fused conv-bias-relu is used in classification backbones that put ReLU *before* BN or that skip BN. YOLOv1 as implemented here is conv–BN–leaky.

### 4.11.2 `spatial_moments_kernel`

One CUDA block per channel (`blockIdx.x = channel`). Threads stride through `batch * H * W` elements of that channel, accumulate `sum` and `sum_sq` in registers, then tree-reduce in shared memory (`kMomentThreads`). Thread 0 writes:

```text
mean[c] = sum / count
var[c]  = max(sum_sq/count - mean², 0)
```

NCHW indexing is `((n * C + c) * spatial) + hw`. The kernel is templated on `Act` (`float` or `__half`) with `load_act` converting to float for the accumulation so FP16 activations still produce FP32 moments.

### 4.11.3 `finalize_bn_stats_kernel`

One thread per channel. In training:

```text
inv_std[c] = rsqrt(max(var[c] + eps, kSafeEps))
running_mean = (1-m) * running_mean + m * batch_mean
running_var  = (1-m) * running_var  + m * batch_var
```

In eval, `mean` and `inv_std` are overwritten from the running statistics. This kernel is tiny (`C` is at most 1024 in YOLO) and is not the bottleneck; it exists so the fused map does not recompute `rsqrt` per spatial location.

### 4.11.4 `fused_bn_leaky_kernel`

```cuda
channel = (index / spatial) % channels
xhat    = (x[index] - mean[c]) * inv_std[c]
bn      = gamma[c] * xhat + beta[c]
out     = bn > 0 ? bn : bn * slope
```

Broadcast loads of 1D `mean`, `inv_std`, `gamma`, `beta` hit cache. Compared to `BatchNorm2d::forward` then `LeakyReLU::forward`, this kernel eliminates a full NCHW global-memory round trip. At 448×448×64 that round trip is tens of mebibytes per block; across the YOLO stem it is measurable in Nsight.

### 4.11.5 Backward

`leaky_backward_from_output_kernel` uses the *fused output* sign (`activated > 0`) rather than storing a separate mask tensor: `d_bn = d_y * (activated > 0 ? 1 : slope)`. Then BN backward (gamma/beta grads, `dxhat`) and cuDNN convolution backward write into `ensure`d caches `grad_bn_cache_` and `grad_conv_cache_`. Parameters (`conv` weights/bias, `gamma_`, `beta_`) are updated with `sgd_update_` (same kernel as FC).

### 4.11.6 Why not a handwritten convolution

A student NCHW implicit-GEMM would lose cuDNN’s Winograd / implicit GEMM / Tensor Core selection, which *is* LibTorch’s backend. The thesis fuses **around** cuDNN. NVIDIA owns convolution performance; the student owns the glue that LibTorch also glues but hides.

## 4.12 `YOLOLoss` forward, backward, and mean — one design

### 4.12.1 Grid layout

Predictions and targets are `[B, 7, 7, 10+C]` or flattened `[B, 7*7*(10+C)]`. `as_yolo_grid` views either as 4D. Each cell holds two boxes of five numbers `(x, y, w, h, conf)` plus `C` class scores (`CLASS_OFFSET = 10`).

### 4.12.2 `yolo_loss_forward_kernel`

One thread per cell (`cell_count = B * 49`). The thread:

1. Computes IoU of both predicted boxes against the target box (`yolo_iou` device function).
2. Selects the responsible box (higher IoU).
3. Accumulates localisation (xy, sqrt-wh), objectness, no-object, and classification terms with YOLOv1 λ constants (`λ_coord = 5`, `λ_noobj = 0.5` in the original paper; the source uses the same structure).
4. Writes a **scalar cell loss** into `cell_loss[cell]`.

There is no host branch on objectness. The object mask is data. Training therefore does not pull a 7×7 mask to the CPU.

### 4.12.3 `yolo_mean_loss_kernel` (the Thrust replacement)

```cuda
__global__ void yolo_mean_loss_kernel(const float* cell_loss, float* mean_loss,
                                      int cell_count, float inv_batch)
{
    __shared__ float shared_sum[kThreads];
    float partial = 0.0F;
    for (int index = threadIdx.x; index < cell_count; index += blockDim.x)
        partial += cell_loss[index];
    shared_sum[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        mean_loss[0] = shared_sum[0] * inv_batch;
}
```

Launched as `<<<1, kThreads, 0, stream>>>`. For `B=16`, `cell_count=784`, one block is enough; the grid-stride loop in the partial sum handles larger B. The output is a GPU tensor of shape `[1]`.

**Historical bug.** `thrust::reduce` on a `device_ptr` returns a host `float`. That return is a device-wide synchronisation. Calling it once per batch inside `YOLOLoss::loss` stalled the pipeline *in addition to* the logger’s `to_host`. After the change, `loss()` returns a 1-element GPU tensor; the training loop `to_host`s it when printing—one sync per printed batch.

Workspace is process-static (`YoloWorkspace` with `optional<Tensor> cell_loss, grad, scalar`). `ensure` means YOLO loss allocates at most once per unique `(B, C)` pair.

### 4.12.4 Backward

`yolo_loss_backward_kernel` writes `dL/dpred` with the same responsible-box logic, scaled by `inv_batch`. Mixed precision then multiplies by `loss_scale()` if AMP is on. Classification terms use `2 * (pred - tgt) * obj_mask * inv_batch` (MSE on class scores, as in YOLOv1). The gradient tensor is `ensure`d and returned as a view.

Standalone `calculate_iou` launches `yolo_iou_kernel` for `[N,4]` boxes. Training uses the inlined IoU inside the cell kernel to avoid an extra global buffer of IoUs.

## 4.13 NVTX and remaining milliseconds

Hot functions wrap `dl::NvtxRange("FullyConnected_Forward")` and similar. Nsight Systems then shows Custom FC versus Torch FC on the same timeline. After fusion, the remaining epoch time is attributed to GEMM, conv, loss, and H2D—not to `cudaMalloc` or `thrust::reduce`.

After fused SGD, GPU loss reduce, `matmul_into`, TF32, and cache reuse, measured Custom versus Torch YOLO training at batch 16 sat at approximately **77.8 ms versus 76.4 ms** per epoch on the development RTX-class GPU, with Custom forward sometimes slightly faster than Torch (20.5 vs 23.5 ms in one micro pass) and FC backward comparable (1.96 vs 2.24 ms). Those numbers move with driver and clocks; Chapter 6 describes how to reproduce them. The architectural point is that the remaining gap is **not** an allocator gap and **not** a Thrust-reduce gap. Hypothetical further wins (NHWC layout, CUDA Graphs) were estimated at ~5–10% and were left unimplemented so the thesis can defend a finished, auditable stack rather than an unbounded optimisation backlog.

## 4.14 Launch hygiene (normative)

All custom kernels pass `current_stream()` as the fourth launch parameter. `cudaGetLastError()` is checked after launch. Layers do not call `cudaDeviceSynchronize()`. The only intended syncs are CUDA events in `Profiler::stop`, `to_host`, and `cudaStreamSynchronize` on a **copy** stream before compute consumes that slot (Chapter 5).
