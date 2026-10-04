# Tensor.cpp

Tensor memory, views, element-wise arithmetic on the GPU, and matrix multiplication through cuBLAS.

Kernels, cuBLAS, the stream, and the pinned pool are compiled only when `DEEPLEARNLIB_ENABLE_CUDA` is set. Methods that need them throw without that flag.

## calculate_size

Product of the dimensions. The accumulator starts at 1, so an empty shape has size 1 (a scalar).

## make_contiguous_strides

Contiguous row-major strides. The last dimension gets 1, and each earlier stride is the product of the stride and the length of the dimension on its right. An empty shape returns an empty stride vector.

## infer_view_shape

Copies `new_shape`. At most one `-1`; a second one, or a dimension < 0, throws. When there is a `-1`, it writes `numel / product_of_known`. A known product of 0 with a non-zero `numel` throws; with a zero `numel` it writes 0. A product that does not divide `numel` throws. Without `-1`, the product must equal `numel`.

## conversion_launch

A grid of `(count + 255) / 256` blocks. The caller supplies the block size (`kInplaceThreads` = 256).

## add_inplace_f32_kernel

`dst[i] += src[i]`, one thread per element. Threads past `count` do nothing.

## mul_inplace_f32_kernel

`dst[i] *= scalar`.

## mul_into_f32_kernel

`out[i] = lhs[i] * rhs[i]`. The result goes to a separate buffer.

## clamp_inplace_f32_kernel

Clamps each element to `[lo, hi]`.

## add_scaled_inplace_f32_kernel

`dst[i] += scale * src[i]`. A negative scale subtracts without a second kernel.

## sgd_update_f32_kernel

`update = grad + decay * weight`. When `clip > 0`, clamps the update to `[-clip, clip]`. Then `weight -= lr * update`.

## sgd_momentum_update_f32_kernel

The same update and the same clamp as SGD. `velocity = momentum * velocity + update`, then `weight -= lr * velocity`. The velocity stays in its buffer.

## add_row_f32_kernel

`dst[i] += bias[i % features]`. The same bias repeats on every row of a `[batch, features]` matrix.

## add_sum_rows_f32_kernel

One thread per column. Sums that column across rows and writes `dst[col] = beta * dst[col] + sum`.

## add_scalar_f32_kernel

`dst[i] += scalar`.

## transpose_2d_f32_kernel

`output[column * rows + row] = input[row * cols + column]`. The result is in a new buffer.

## sum_f32_kernel

A grid that loops over the elements. The block reduces its partial sum in shared memory (256 threads), and thread 0 does `atomicAdd` into `out`.

## any_nonfinite_f32_kernel

When an element is not finite, writes `1` through `flag`. No atomic: it is enough that anyone stores a one.

## CublasContext::CublasContext

`cublasCreate`, then `CUBLAS_TF32_TENSOR_OP_MATH` (tensor cores, a TF32 mantissa instead of full fp32). Allocates a 64 MiB workspace and passes it to the handle through `cublasSetWorkspace`.

## CublasContext::~CublasContext

`cublasDestroy` and `cudaFree` when the pointers are not null. Error codes are discarded, because this runs as the process shuts down.

## CublasContext::handle

A static `CublasContext` per process. Returns its `cublasHandle_t`.

## get_cublas_handle

Shorthand for `CublasContext::handle`. GEMM calls use this.

## current_stream

Returns the thread's `thread_local` stream. Zero is the default stream.

## set_current_stream

Stores the thread's stream. cuBLAS is updated only by `StreamGuard`.

## StreamGuard::StreamGuard

Remembers `current_stream()`, replaces it with the given stream, and calls `cublasSetStream` on that same stream. Kernels and GEMM inside the guard's scope run together.

## StreamGuard::~StreamGuard

Restores the previous thread stream and the cuBLAS stream. The destructor also runs when the body throws.

## Tensor::Tensor()

Delegates to the shape constructor with an empty vector, `Device::CPU`, and `Dtype::Float32`. The product of an empty shape stays 1, so this is a scalar.

## Tensor::Tensor(shape, device, dtype)

Computes the size and the strides. On the GPU, checks that there is at least one device, calls `cudaMalloc`, and stores the buffer in a `shared_ptr` with `CudaDeleter`. `cudaMalloc` does not zero the memory. On the CPU, uses `operator new`, zeroes it with `memset`, and attaches `CpuDeleter`. Without CUDA, a GPU device throws.

## Tensor::Tensor(shape, strides, data, device, dtype)

A view: takes the shape, the strides, and the existing `shared_ptr`. It does not allocate and does not check the strides. The contiguity check is in `view()`, before this constructor.

## Tensor::get_shape

Returns `shape_`.

## Tensor::get_strides

Returns `strides_`.

## Tensor::get_size

Returns `size_`.

## Tensor::get_device

Returns `device_`.

## Tensor::get_dtype

Returns `dtype_`.

## Tensor::element_size

Returns `dl::element_size(dtype_)`.

## Tensor::nbytes

`size_ * element_size()`.

## Tensor::get_data

Returns `data()` as `const float*`.

## Tensor::data

Returns `data_.get()`, the mutable and const overloads. Does not copy the buffer.

## Tensor::describe

Joins `format_shape`, `dtype_name`, the text `GPU` or `CPU`, and `n=` with the element count. Used in error messages.

## Tensor::to_dtype

The stream argument is unused. A `dtype` other than its own throws. The same dtype returns `view(shape_)`.

## Tensor::compute_strides

Writes the result of `make_contiguous_strides(shape_)` into `strides_`.

## Tensor::is_contiguous

An empty shape returns true. From the back, checks that the stride equals the expected row-major stride. A dimension of length 1 may have any stride. The expected stride is multiplied by the dimension length.

## Tensor::ensure_gpu

Throws when the device is not GPU, or when `size_ > 0` and the pointer is null. `op_name` is included in the message.

## Tensor::ensure_binary_op

Both tensors go through `ensure_gpu`. Requires equal size, contiguity of both, and the same dtype.

## plan_rowmajor_gemm

Builds a `GemmPlan` (`M`, `N`, `K`, `lda`, `ldb`, `ldc`, transpose flags, result shape). Both tensors must be on the GPU, have a pointer, a non-empty shape, and the same dtype. A transpose requires rank 2.

`M` is the columns of A when `transpose_a` is set, otherwise `size(A) / last dimension`. `K` is the rows of A when transposed, otherwise the last dimension of A. `other_k` is the columns of B when `transpose_b` is set, otherwise the first dimension of B. `N` is the rows of B when transposed, otherwise `size(B) / other_k`. `K` must equal `other_k` and be positive.

Result shape: with `transpose_a`, appends `a_shape[1]`, otherwise every dimension of A except the last; with `transpose_b`, appends `b_shape[0]`, otherwise the dimensions of B except the first. An empty shape becomes `{1}`.

cuBLAS is column-major. Row-major `C = A*B` is column-major `C^T = B^T * A^T`, so `trans_a` receives the flag from `transpose_b`, and `trans_b` receives the flag from `transpose_a`. `lda` is `b_shape[1]` when B is transposed, otherwise `N`. `ldb` is the last dimension of A. `ldc` is `N`.

## launch_rowmajor_gemm

Returns immediately when `M` or `N` is 0. Sets the cuBLAS stream to `current_stream()`. `cublasGemmEx` receives matrix B first, then A, and the dimensions `N, M, K` — the same swap as in the plan. Alpha is 1, beta comes from the argument. Storage for A, B, and C is `CUDA_R_32F`. The compute type is `CUBLAS_COMPUTE_32F_FAST_TF32`: the multiply runs on tensor cores, and the accumulation stays fp32.

## Tensor::matmul

Without CUDA, throws. Requires contiguous operands. Computes the plan, allocates a result of `result_shape`, and calls GEMM with beta 0 (overwrite from zero).

## Tensor::matmul_into

Without CUDA, throws. The same contiguity requirements. `out` must be on the GPU, with the operand dtype and the shape from the plan. Beta scales the current contents: 0 overwrites, 1 adds. Returns `out`.

## Tensor::ensure

When the slot is empty, or the shape, device, or dtype does not match, stores a new `Tensor`. Otherwise returns the existing one. The training loop therefore does not call `cudaMalloc` on every step.

## Tensor::operator+(const Tensor&)

Without CUDA, throws. `ensure_binary_op`. A new tensor of the same shape; a size of 0 returns it immediately. Otherwise copies the left-hand side device-to-device and calls `add_`.

## Tensor::operator-(const Tensor&)

Like the sum, but the right-hand side goes through `add_scaled_` with scale `-1`.

## Tensor::operator*(const Tensor&)

`ensure_binary_op`, a new tensor, and `mul_into` when the size is non-zero. Without CUDA, throws.

## Tensor::operator*(float)

Without CUDA, throws. The tensor must be on the GPU and contiguous. Copy, then `mul_(scalar)`. Size 0 returns an empty result without a copy.

## Tensor::operator+(float)

Like scalar multiplication, but after the copy it launches `add_scalar_f32_kernel` on the current stream.

## Tensor::add_

`ensure_binary_op`, and `add_inplace_f32_kernel` when the size is non-zero. Returns `*this`. Without CUDA, throws.

## Tensor::mul_

GPU, contiguous tensor, then `mul_inplace_f32_kernel`. Without CUDA, throws.

## Tensor::mul_into

`ensure_binary_op`. `out` must be on the GPU, with the same dtype and size. Kernel `mul_into_f32_kernel`. Returns `out`. Without CUDA, throws.

## Tensor::add_scaled_

`ensure_binary_op` and `add_scaled_inplace_f32_kernel`. Without CUDA, throws.

## Tensor::sgd_update_

`ensure_binary_op` with the gradient and `sgd_update_f32_kernel`. The weights live in `*this`. Without CUDA, throws.

## Tensor::sgd_momentum_update_

`ensure_binary_op` for the gradient and for `velocity`, then `sgd_momentum_update_f32_kernel`. Without CUDA, throws.

## Tensor::add_row_

GPU, bias on the GPU, rank 2, both contiguous, the same dtype, bias size equal to the feature count. The kernel adds the bias to every row. Without CUDA, throws.

## Tensor::add_sum_rows_

GPU, rank-2 matrix, both contiguous, the same dtype. The length of `*this` must equal the column count. The kernel implements `this[j] = beta * this[j] + sum of the rows of column j`. Without CUDA, throws.

## Tensor::clamp

GPU, contiguous, `lo <= hi`. Copy and `clamp_`. Without CUDA, throws.

## Tensor::clamp_

The same conditions, clamped in place by `clamp_inplace_f32_kernel`. Without CUDA, throws.

## Tensor::has_non_finite

Size 0 returns false. On the GPU (when CUDA is enabled) it keeps a static flag from `cudaMalloc`, zeroes it on the current stream, launches `any_nonfinite_f32_kernel`, copies the flag to the host, and synchronizes the stream — the read is valid only after the kernel and the memcpy. Returns whether the flag is non-zero. On the CPU it walks the buffer with `std::isfinite`.

## Tensor::assert_finite

With `DEBUG_NUMERICS`, throws `"NaN detected in ..."` when `has_non_finite` is true. A null `context` is replaced with the word `Tensor`. Without that macro the function does nothing.

## Tensor::sum

Without CUDA, throws. GPU, contiguous; a `dim` other than `-1` throws (an axis is not implemented). The result is an fp32 scalar zeroed with `cudaMemsetAsync`. When `size_ > 0`, launches `sum_f32_kernel`, at most 1024 blocks.

## Tensor::view

Requires `is_contiguous`. Shape from `infer_view_shape`, strides from `make_contiguous_strides`, the same `shared_ptr`. A new view, with no copy of the data.

## Tensor::as_view

`view(shape_)`. The same shape, a shared buffer, without giving away exclusive ownership.

## Tensor::transpose

Without CUDA, throws. GPU, rank 2, contiguous. The result has shape `{columns, rows}`. An empty dimension returns it without a kernel. Otherwise `transpose_2d_f32_kernel`.

## Tensor::zeros_like

Constructs a tensor with the pattern's shape, device, and dtype. The CPU constructor already zeroes. On the GPU (when CUDA is enabled) and with a non-zero size, it also calls `cudaMemsetAsync` on the current stream, because `cudaMalloc` does not clear memory.

## PinnedPool::~PinnedPool

For the four slots, destroys the event and calls `cudaFreeHost` when they exist. Errors are discarded.

## PinnedPool::acquire

Takes the next slot in a cycle (`kPinnedSlots` = 4). When the slot has a recorded event, synchronizes it — the host waits until the previous memcpy has finished reading the buffer. When the capacity is too small, frees the old block and `cudaMallocHost`s a new size. Creates the event with `cudaEventDisableTiming` if it did not exist. Returns the pointer.

## PinnedPool::record

Finds the slot with that pointer, records its event on the given stream, and sets `recorded`. The next `acquire` therefore does not overwrite the buffer before the memcpy finishes.

## pinned_pool

A static `PinnedPool` per process.

## Tensor::to_host

A vector of length `size_`. Size 0 returns it immediately. A null pointer throws. On the GPU, copies through a pinned slot (`cudaMemcpyAsync` D2H), records the event, and synchronizes the stream — the caller needs a host `float` vector, and that vector may be filled only after the copy arrives. On the CPU, `std::copy`.

## Tensor::from_host(shape, vector)

The product of the dimensions must equal `host_data.size()`. An empty shape does not enter the loop, so the expected size stays 1. Then calls the pointer overload.

## Tensor::from_host(shape, pointer)

Creates an fp32 tensor. Size 0 returns it immediately, and for another `dtype` calls `to_dtype` (which still accepts only the current type). A null `host_data` with a non-zero size throws. On the GPU, copies into a pinned slot and does an asynchronous H2D; `record` protects the slot, and the function does not synchronize the host. On the CPU, ignores the stream and copies with `std::copy`.
