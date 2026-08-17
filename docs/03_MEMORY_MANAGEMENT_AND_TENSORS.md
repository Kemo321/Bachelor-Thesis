# Chapter 3 — Memory management and `dl::Tensor`

## 3.1 Role of the tensor

`dl::Tensor` is the only device array type on the Custom training path. Layers do not call `cudaMalloc` themselves for activations. They either:

- construct a `Tensor` (rare after warmup), or
- call `Tensor::ensure` on an `std::optional<Tensor>` member so that a matching shape **reuses** the previous allocation.

LibTorch presents a similar illusion (`at::Tensor` with a caching allocator). DeepLearnLib makes the cache **lexical and local**: you can grep `ensure(` and list every buffer that survives across steps. That grep is the thesis's memory budget.

## 3.2 Allocation lifecycle: `cudaMalloc` and `CudaDeleter`

GPU storage is a `std::shared_ptr<float>` whose deleter is `dl::CudaDeleter`:

```cpp
struct CudaDeleter {
    void operator()(float* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};
```

The constructor path in `Tensor.cpp` (device, non-empty) computes `bytes = size * element_size()`, then:

```text
CHECK_CUDA(cudaMalloc(&gpu_pointer, bytes));
data_ = std::shared_ptr<float>(gpu_pointer, CudaDeleter{});
```

(The pointer is typed `float*` even for FP16 storage; `half_data()` reinterprets. This is an implementation convenience, not a claim that FP16 elements are 4 bytes. `nbytes()` uses `element_size()`.)

**Ownership rules:**

- Copy construction of `Tensor` is **deleted**. Accidental copies would either double-free (if unique_ptr) or hide D2D clones (if deep copy). Neither is acceptable on a training hot path.
- Move construction is defaulted. Returning a Tensor from `forward` is a move of the wrapper, not a clone of VRAM, provided the callee returns a view or a cache alias.
- `view` / `as_view` construct a new wrapper with the **same** `shared_ptr`. The last wrapper to die runs `cudaFree`.

CPU tensors use `CpuDeleter` (`operator delete`). Training tensors are GPU. Host staging for H2D uses pinned memory (`cudaMallocHost`) behind `PinnedHostDeleter`, which synchronises the associated stream before `cudaFreeHost` so that an in-flight DMA cannot outlive the staging buffer.

## 3.3 Error handling around the driver

Every runtime call on this path is wrapped:

```text
CHECK_CUDA(...)   → dl::check_cuda  → log + throw std::runtime_error
CHECK_CUBLAS(...) → dl::check_cublas
CHECK_CUDNN(...)  → dl::check_cudnn
```

Failures are not `assert`. A thesis binary that hits `cudaErrorMemoryAllocation` must unwind with a message that includes file and line, because the next question is always “which `ensure` or GEMM workspace overflowed.”

## 3.4 The memory–time tradeoff

### 3.4.1 Statement

DeepLearnLib **deliberately over-allocates** relative to a naive “allocate output, free after the next layer” scheme. For YOLOv1 training at batch size 16 and 448×448 inputs, the process resident VRAM is on the order of **~13 GiB** on the development GPU (read from `Profiler::get_vram_usage_mb()`, which is `cudaMemGetInfo`: used ≈ total − free). That number is not a kernel occupancy peak; it is **process-visible** memory, including:

- every layer's `output_cache_`, `input_cache_`, `grad_input_cache_`;
- convolution algorithm workspaces retained by cuDNN descriptors;
- the **64 MiB** persistent cuBLAS workspace (`kCublasWorkspaceBytes`);
- YOLOLoss static workspace (`cell_loss`, `grad`, `scalar`);
- the current (and, with double buffering, *next*) uploaded batch;
- CUDA context and library internals.

The alternative—`cudaMalloc` / `cudaFree` per layer per step—looks “lean” in a profiler screenshot and is **catastrophic** in wall-clock time. The CUDA allocator takes a device-wide lock; `cudaFree` often synchronises. LibTorch's caching allocator exists because this lesson was learned at industry scale. DeepLearnLib reproduces the lesson with a simpler mechanism: **typed optional members**.

### 3.4.2 `Tensor::ensure`

```cpp
auto Tensor::ensure(std::optional<Tensor>& slot,
                    const std::vector<int>& shape,
                    Device device,
                    Dtype dtype) -> Tensor&
{
    if (!slot.has_value()
        || slot->get_shape() != shape
        || slot->get_device() != device
        || slot->get_dtype() != dtype)
    {
        slot = Tensor(shape, device, dtype);  // cudaMalloc once
    }
    return *slot;
}
```

After the first forward of a given shape, `slot->get_shape() == shape` and the function is a pointer return. There is no cudaMalloc, no free, no stream sync.

If the user changes batch size at runtime, `ensure` reallocates. The thesis pipelines freeze `batch_size` from `experiments.json` for this reason.

### 3.4.3 What each layer caches (FullyConnected as exemplar)

```text
input_cache_        view or dtype-converted input   (needed for dW = X^T dY)
output_cache_       Y = X W                         (reused every step)
grad_input_cache_   dX = dY W^T                     (returned to previous layer)
weights_            [in, out]                       (parameters)
biases_             [1, out]
weights_gradient_
biases_gradient_
```

`forward` writes:

```cpp
input_cache_ = input_tensor.as_view();   // no D2D if dtype matches
Tensor& output = Tensor::ensure(output_cache_, {B, out}, GPU, dtype);
input_cache_->matmul_into(weights_, output);
output.add_row_(biases_);
return output.as_view();
```

`as_view()` on the return prevents the caller from believing they own a unique buffer; the next forward will overwrite `output_cache_`. Callers that need to retain Y (they should not, except the loss) must copy explicitly—and the training loop does not.

Conv2d, MaxPool2d, Dropout, Softmax, LeakyReLU, BatchNorm2d, FusedCBR2d follow the same pattern. Grep `ensure(` in `src/` is the complete cache inventory.

### 3.4.4 Why ~13 GiB is acceptable

The comparison target is LibTorch, which also holds activations for backward and also caches allocator blocks. A Custom process that used 3 GiB but spent 40% of the epoch in `cudaMalloc` would *lose* the thesis comparison. A Custom process that uses 13 GiB and GEMMs at TF32 Tensor Core rates can match Torch's ~76 ms/epoch. **Memory is spent to buy time.** The documentation must not describe this as a bug.

Inference-only future work could drop `input_cache_` after `eval()`; that is not the training thesis.

## 3.5 Streams, cuBLAS handle, and the 64 MiB workspace

`CublasContext` is a process-wide Meyers singleton:

```cpp
CHECK_CUBLAS(cublasCreate(&handle_));
CHECK_CUBLAS(cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
CHECK_CUDA(cudaMalloc(&workspace_, 64 * 1024 * 1024));
CHECK_CUBLAS(cublasSetWorkspace(handle_, workspace_, 64 * 1024 * 1024));
```

Without a persistent workspace, `cublasGemmEx` may allocate scratch per call—the exact anti-pattern `ensure` exists to prevent, one level down the stack. TF32 math mode allows Ampere/Hopper/Blackwell Tensor Cores on FP32 inputs; accuracy is acceptable for YOLO training and matches Torch's common `allow_tf32` behaviour when cuDNN benchmark mode is on.

`cublasSetStream(handle, current_stream())` is called immediately before each GEMM so that double-buffered training (Chapter 5) actually overlaps.

`StreamGuard` pushes `current_stream()` for a lexical scope and pops in the destructor. Layers bind cuDNN to the same stream via `bind_cudnn_stream`. Nested guards are well-defined because the previous stream is stored on the guard object, not in a global stack beyond that.

## 3.6 Zero-allocation GEMM: `matmul_into` and logical transposes

### 3.6.1 The row-major / column-major mismatch

DeepLearnLib stores matrices **row-major**, as do NumPy and PyTorch. cuBLAS interprets pointers as **column-major**. The identity that makes this cheap is:

```text
row-major(A)  ≡  column-major(A^T)
```

Therefore a row-major product `C = A @ B` can be issued as a column-major product of the swapped operands with swapped dimensions. `plan_rowmajor_gemm` encodes:

```text
trans_a (cuBLAS) = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N
trans_b (cuBLAS) = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N
lda, ldb, ldc    derived from N, K, and the logical flags
```

`cublasGemmEx` is then launched as `(N, M, K)` with pointers `(B, A, C)`—the textbook “compute C^T = B^T A^T” trick—without ever writing a transposed clone of `A` or `B`.

### 3.6.2 Why physical `.transpose()` is forbidden on the hot path

A naive backward for `Y = X W` is:

```text
dW = transpose(X) @ dY
dX = dY @ transpose(W)
```

If `transpose` allocates a new tensor and copies, each FullyConnected layer pays two D2D copies of enormous buffers. For the YOLO head, `X` is `[B, 7*7*1024]` and `W` is `[7*7*1024, 4096]`. Copying `W` alone is `7*7*1024*4096*4 ≈ 804 MiB`. At a realistic device copy bandwidth this is **tens of milliseconds**, and across many FC-like GEMMs in a naive port it was measured on the order of **~95 ms per layer** of wasted bandwidth. Logical `CUBLAS_OP_T` reduces that cost to **zero extra bytes** and whatever cuBLAS already reads.

### 3.6.3 `matmul` versus `matmul_into`

`matmul` allocates `Tensor result(plan.result_shape, GPU, dtype)` and is used when the caller has no cache (tests, one-shot scripts).

`matmul_into` requires `out` to already have `plan.result_shape` and matching dtype. It writes `C = op(A) op(B) + beta * C`. FullyConnected uses:

| Call | Meaning | beta |
| --- | --- | --- |
| `X.matmul_into(W, Y)` | `Y = X W` | 0 |
| `X.matmul_into(dY, dW, transA=true, transB=false, beta=inertia_)` | `dW = X^T dY + μ dW` | momentum |
| `dY.matmul_into(W, dX, transA=false, transB=true)` | `dX = dY W^T` | 0 |

No GEMM output allocation after warmup.

FP32 GEMM uses `CUBLAS_COMPUTE_32F_FAST_TF32`. FP16 GEMM uses `CUBLAS_COMPUTE_32F` accumulation with `CUDA_R_16F` pointers and `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

## 3.7 In-place elementwise API

Allocating `operator+` still exists and, in the current tree, is implemented with Thrust `transform` into a **new** tensor. That path is for tests and rare graph edges. The training path uses:

| Method | Semantics | Allocation |
| --- | --- | --- |
| `add_` | `this += other` | none |
| `mul_` | `this *= scalar` | none |
| `add_scaled_` | `this += scale * other` | none |
| `clamp_` | clamp in place | none |
| `mul_into` | `out = this * other` | none (out preallocated) |
| `add_row_` | broadcast-add bias | none |
| `add_sum_rows_` | `bias_grad = beta * bias_grad + sum_rows(dY)` | none |
| `sgd_update_` | fused decay + clip + add | none |

`sgd_update_` is:

```text
u = grad + decay * w
u = clip(u, [-clip, clip])   if clip > 0
w = w - lr * u
```

in one kernel (`sgd_update_f32_kernel` / `_f16_kernel`). Weight decay is **not** applied in `backward()`, which would have required extra buffers and an extra launch.

## 3.8 Host traffic

`to_host(stream)` copies to `std::vector<float>` and **synchronises** that stream. It is allowed for:

- logging a scalar loss (and `Network::fit` only does this when `verbose`);
- mAP decoding on VOC custom;
- writing checkpoints (`save` reads parameters to host).

It is forbidden inside `Layer::forward` / `backward` and inside `YOLOLoss::loss` except that the *caller* may `to_host` the returned 1-element mean. The mean itself is produced by `yolo_mean_loss_kernel` on device (Chapter 4).

`from_host` is the dataloader upload. It may use pinned staging. It must be given the compute stream that will consume the batch, or a dedicated copy stream in the double-buffer scheme.

## 3.9 Views versus materialisation

`view(new_shape)` shares storage if the tensor is contiguous and the product of dimensions matches. Flatten is a view from NCHW to `[B, C*H*W]` when possible. `as_view()` clones the wrapper only.

A D2D copy (`memcpy_d2d_on_current`) is used when dtype conversion or a true clone is required. Skipping those copies on the input of every layer (passing `as_view` of the previous cache) was a measured win: it removes a full activation copy per layer from the epoch.

## 3.10 Mixed precision storage

`Dtype::Float16` stores `__half`. Host I/O is still FP32. Layers may convert at the cache boundary (`to_dtype`) and then GEMM in FP16. Loss scale is applied in `YOLOLoss::loss_derivative` when `dl::loss_scale() != 1`. Default thesis configs in `experiments.json` keep `precision: fp32` so that Custom versus Torch comparisons are not confounded by AMP policy differences unless mixed precision is the variable under study.

## 3.12 The `cudaMalloc` path in `Tensor.cpp`

The device constructor (non-empty GPU tensor) is the only place Custom training is allowed to call `cudaMalloc` for activations. Layers must not call it. The sequence is:

1. Compute `bytes = size * element_size()`. FP32 uses 4; FP16 uses 2. The `shared_ptr` is still typed `float*` for historical reasons; `half_data()` reinterprets. `nbytes()` is the source of truth for copy sizes.
2. `void* gpu_pointer = nullptr;` then `CHECK_CUDA(cudaMalloc(&gpu_pointer, bytes));`
3. `data_ = std::shared_ptr<float>(static_cast<float*>(gpu_pointer), CudaDeleter());`

`CudaDeleter::operator()` is:

```cpp
void operator()(float* ptr) const
{
    if (ptr)
        cudaFree(ptr);
}
```

`cudaFree(nullptr)` is documented as a no-op; the null check is belt-and-braces and makes CPU/GPU deleters look alike. The deleter does **not** call `cudaDeviceSynchronize`. Freeing an in-use buffer is a contract violation of the caller (a view still alive, or a kernel still enqueued). Training avoids that by keeping caches in `std::optional<Tensor>` members that outlive the step.

Empty tensors (size 0) do not call `cudaMalloc`. They exist so that a shape like `{0}` can be moved around in tests without a driver round-trip.

CPU tensors use `CpuDeleter` (`::operator delete`). They are not on the YOLO training path. Host staging for asynchronous H2D uses `cudaMallocHost` and `PinnedHostDeleter`, which **does** synchronise the associated stream before `cudaFreeHost`. That sync is on the *destruction* of a staging buffer, not on every batch: the dataloader reuses pinned buffers where possible. If a staging buffer were destroyed while DMA was in flight, the GPU would read unmapped host memory. The deleter’s sync is the last line of defence for that bug.

## 3.13 Copy, move, and view: why copies are deleted

`Tensor` copy construction is **deleted**. The alternatives were all worse:

- Deep copy on copy-construct would hide a D2D `cudaMemcpy` in `auto t2 = t1;`. Training would accidentally clone activations.
- Shallow copy with a unique_ptr would double-free.
- Shallow copy with shared_ptr *without* deleting the copy constructor would make it too easy to alias caches across layers incorrectly.

The allowed operations are:

- **Move** — defaulted. Returning a Tensor from `forward` moves the wrapper. If the callee returns `output_cache_.as_view()`, the move is a wrapper move of a view that shares `data_` with the cache.
- **`view(shape)`** — new wrapper, same `shared_ptr`, new shape metadata, requires matching product of dimensions and contiguity.
- **`as_view()`** — clone of the wrapper only.
- **`to_dtype`** — may allocate a new buffer if the dtype changes; may return a view if it does not.

Grep for `Tensor t =` in `src/` should not find accidental copies; the compiler would reject them.

## 3.14 Complete `ensure` inventory

The thesis memory budget is the set of `Tensor::ensure` call sites. They are not hidden in a pooling allocator; they are lexical.

| File | Typical slots |
| --- | --- |
| `FullyConnected.cpp` | `output_cache_`, `grad_input_cache_`; weights/biases are constructor-allocated |
| `Conv2d.cpp` | `output_cache_`, `grad_input_cache_`, algorithm workspace via `CudaWorkspace::ensure` |
| `FusedCBR2d.cpp` | `fused_output_cache_`, `bn_input_cache_`, `grad_bn_cache_`, `grad_conv_cache_`, BN vectors |
| `MaxPool2d.cpp` | `output_cache_`, `grad_input_cache_` |
| `BatchNorm2d.cpp` | `output_cache_`, `grad_input_cache_` |
| `LeakyReLU.cpp` | `output_cache_`, `grad_input_cache_` |
| `Dropout.cpp` | `mask_`, `output_cache_`, `grad_input_cache_` |
| `Softmax.cpp` / `Losses.cpp` | probability / row-loss caches |
| `Network.cpp` | `loss_grad_clip_cache_` |
| `YOLOLoss.cpp` | process-static `cell_loss`, `grad`, `scalar` |

After warmup at a frozen batch size, none of these call `cudaMalloc`. Changing `batch_size` in `experiments.json` reallocates on the next `ensure` that sees a new shape. That is why thesis pipelines freeze batch size.

## 3.15 Why ~13 GiB is the correct YOLO number to publish

`Profiler::get_vram_usage_mb()` is `(total - free) / 2^20` from `cudaMemGetInfo`. It is process-visible memory, not a kernel occupancy peak. For YOLOv1 train at batch 16, 448×448, FP32, the development GPU has shown on the order of **13 GiB** after caches are warm. That figure includes:

- activations for every FusedCBR block at 448, 224, 112, 56, 28, 14, 7 spatial sizes, NCHW, plus backward caches;
- two 4096-wide FC layers and their `dW` buffers (`7*7*1024 × 4096 × 4 bytes ≈ 804 MiB` for `W` alone, plus `dW`);
- cuDNN convolution workspaces retained after `cudnnFindConvolutionForwardAlgorithm`;
- 64 MiB persistent cuBLAS workspace;
- YOLOLoss `cell_loss[B*49]`, `grad` of prediction shape, scalar;
- current and next uploaded batch (double buffer, Chapter 5);
- CUDA context, cuDNN handle, driver allocations.

A “theoretical minimum” that counts only weight tensors is perhaps 1–2 GiB and is **not** comparable to LibTorch, which also holds activations for backward and also caches allocator blocks. Publishing 13 GiB is honesty. Calling it a leak is incorrect if the curve is **flat** after epoch 1 (Chapter 6).

Inference-only future work could drop `input_cache_` after `eval()`. That is not this thesis.

## 3.16 Zero-allocation GEMM: the algebra in full

DeepLearnLib stores matrices row-major. cuBLAS interprets pointers as column-major. For any matrix `M`,

```text
row_major_ptr(M)  ≡  column_major_ptr(M^T)
```

with leading dimension equal to the row-major number of columns.

The product we want is the row-major product `C = A @ B` with `A` ∈ R^{M×K}, `B` ∈ R^{K×N}, `C` ∈ R^{M×N}. Taking transposes,

```text
C^T = B^T @ A^T
```

in column-major arithmetic. cuBLAS `cublasGemmEx` computes

```text
C_col = α op(X) op(Y) + β C_col
```

with `op` either identity or transpose, dimensions `(m, n, k)` meaning `op(X)` is m×k and `op(Y)` is k×n. If we pass **pointers** `(X, Y, C) = (B, A, C)` and dimensions `(m, n, k) = (N, M, K)`, we are computing an N×M result which *is* `C^T` in column-major, i.e. `C` in row-major. Logical transposes of the *row-major* operands then swap which cuBLAS `CUBLAS_OP_T` is set:

```text
plan.trans_a = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;  // applied to pointer B
plan.trans_b = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;  // applied to pointer A
plan.lda     = transpose_b ? b_shape[1] : plan.N;
plan.ldb     = a_shape.back();
plan.ldc     = plan.N;
```

This is `plan_rowmajor_gemm` in `Tensor.cpp`. No temporary `A^T` buffer is ever allocated.

### 3.16.1 FullyConnected mapping

For `Y = X W` with `X` `[B, in]`, `W` `[in, out]`, `Y` `[B, out]`:

```text
X.matmul_into(W, Y)                         // Y = X W,           beta = 0
X.matmul_into(dY, dW, true, false, μ)       // dW = X^T dY + μ dW
dY.matmul_into(W, dX, false, true)          // dX = dY W^T
```

The middle call is the reason `beta` exists. Momentum (`inertia_`, 0.9 on YOLO FC layers) is fused into the GEMM: cuBLAS writes `dW := X^T dY + μ dW` without a separate `axpy` kernel or a cloned `dW` buffer.

### 3.16.2 The ~95 ms physical transpose tax

A naive port that implemented `dW = transpose(X) @ dY` by allocating `X^T` would copy `B × in` floats, and `dX = dY @ transpose(W)` would copy `in × out` floats. For the YOLO head, `W` is `[7*7*1024, 4096]`:

```text
7 × 7 × 1024 × 4096 × 4 bytes ≈ 804 MiB
```

At a realistic device memcpy bandwidth of ~1 TB/s on a modern RTX-class GPU that copy is on the order of a millisecond; on earlier measurements with additional transposes in the graph and lower effective bandwidth (uncoalesced naive kernels, or PCIe-bound mistakes), the tax was recorded on the order of **~95 ms per layer**. Logical `CUBLAS_OP_T` reduces the extra bytes to **zero**. The remaining time is the GEMM itself, which LibTorch also pays.

`matmul` (allocating) still exists for tests. Training uses only `matmul_into`.

### 3.16.3 Compute types

```text
FP32:  cublasGemmEx(..., CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
FP16:  cublasGemmEx(..., CUDA_R_16F, CUBLAS_COMPUTE_32F,           CUBLAS_GEMM_DEFAULT_TENSOR_OP)
```

TF32 on Ampere and newer matches Torch’s common `allow_tf32` behaviour. Accumulation stays FP32 for FP16 GEMM. `cublasSetStream(handle, current_stream())` is called immediately before each GEMM so double-buffered training actually overlaps.

## 3.17 Persistent 64 MiB cuBLAS workspace

`CublasContext` is a Meyers singleton constructed on first `get_cublas_handle()`:

```text
cublasCreate(&handle_)
cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH)
cudaMalloc(&workspace_, 64 * 1024 * 1024)
cublasSetWorkspace(handle_, workspace_, 64 MiB)
```

Without `cublasSetWorkspace`, `cublasGemmEx` may allocate scratch per call—the exact anti-pattern `ensure` exists to prevent, one level down the stack. 64 MiB is large enough for the YOLO FC GEMMs on the development GPU; if a future layer needed more, the constant `kCublasWorkspaceBytes` is the single place to change. The workspace is process-lifetime, like the handle.

## 3.18 In-place kernels versus allocating `operator+`

`operator+` still uses Thrust `transform` into a **new** tensor. That path is for tests. The training path uses `__global__` kernels in `Tensor.cpp`:

| Kernel | Semantics |
| --- | --- |
| `add_inplace_f32_kernel` / `_f16` | `dst += src` |
| `mul_inplace_*` | `dst *= scalar` |
| `mul_into_*` | `out = lhs * rhs` |
| `clamp_inplace_*` | clamp |
| `add_scaled_inplace_*` | `dst += scale * src` |
| `sgd_update_*` | fused decay + clip + add |
| `add_row_*` | broadcast bias |
| `add_sum_rows_*` | `bias_grad = beta * bias_grad + sum_rows(dY)` |

Launch configuration is `<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>` with `kInplaceThreads` typically 256. The fourth launch parameter is mandatory: a missing stream argument would hit the default stream and serialise with copy streams (Chapter 5).

## 3.19 Host traffic policy

`to_host(stream)` copies to `std::vector<float>` and **synchronises** that stream. Allowed:

- logging a scalar loss when `verbose`;
- mAP decoding on VOC custom;
- writing checkpoints.

Forbidden inside `Layer::forward` / `backward` and inside `YOLOLoss::loss` except that the *caller* may `to_host` the returned 1-element mean. The mean itself is produced on device (Chapter 4).

`from_host` is the dataloader upload. It may use pinned staging. It must be given the compute stream that will consume the batch, or a dedicated copy stream in the double-buffer scheme.

## 3.20 Mixed precision storage

`Dtype::Float16` stores `__half`. Host I/O is still FP32. Layers may convert at the cache boundary (`to_dtype`) and then GEMM in FP16. Loss scale is applied in `YOLOLoss::loss_derivative` when `dl::loss_scale() != 1`. Default thesis configs in `experiments.json` keep `precision: fp32` so that Custom versus Torch comparisons are not confounded by AMP policy differences unless mixed precision is the variable under study.

## 3.21 Contract for layer authors (restated)

1. Own caches as `std::optional<Tensor>`.
2. `ensure` them with the exact shape they will write.
3. Prefer `matmul_into` and `*_` in-place methods.
4. Return `as_view()` of a cache, never a defensive clone.
5. Never `to_host` on the hot path.
6. Never call `cudaMalloc` / `cudaFree` directly.
7. Launch kernels on `current_stream()`.

If those rules hold, VRAM is high, `cudaMalloc` count after warmup is zero, and the remaining time is cuDNN/cuBLAS/custom kernels—the same ingredients LibTorch uses, now visible.
