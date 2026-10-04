#include "DeepLearnLib/Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>

namespace dl
{
namespace
{

    // Product of the dimensions. An empty shape has size 1, so it is a scalar.
    static auto calculate_size(const std::vector<int>& shape) -> int
    {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    }

    // Contiguous row-major strides: the last dimension has stride 1, and each earlier stride is the product of the tail.
    static auto make_contiguous_strides(const std::vector<int>& shape) -> std::vector<int>
    {
        std::vector<int> strides(shape.size());
        if (shape.empty())
        {
            return strides;
        }
        strides.back() = 1;
        for (int dim_idx = static_cast<int>(shape.size()) - 2; dim_idx >= 0; --dim_idx)
        {
            strides[dim_idx] = strides[dim_idx + 1] * shape[dim_idx + 1];
        }
        return strides;
    }

    // Resolves at most one -1, or checks that the product of the dimensions equals numel.
    static auto infer_view_shape(const std::vector<int>& new_shape, size_t numel) -> std::vector<int>
    {
        std::vector<int> shape = new_shape;
        int infer_index { -1 };
        size_t known_product { 1 };

        for (size_t dim_idx = 0; dim_idx < shape.size(); ++dim_idx)
        {
            if (shape[dim_idx] == -1)
            {
                if (infer_index != -1)
                {
                    throw std::runtime_error("view can infer at most one dimension");
                }
                infer_index = static_cast<int>(dim_idx);
            }
            else if (shape[dim_idx] < 0)
            {
                throw std::runtime_error("view shape dimensions must be positive or -1");
            }
            else
            {
                known_product *= static_cast<size_t>(shape[dim_idx]);
            }
        }

        if (infer_index >= 0)
        {
            // A zero axis leaves nothing to divide; only an empty tensor may infer 0.
            if (known_product == 0)
            {
                if (numel != 0)
                {
                    throw std::runtime_error("view cannot infer a dimension when another axis is zero");
                }
                shape[static_cast<size_t>(infer_index)] = 0;
            }
            else if (numel % known_product != 0)
            {
                throw std::runtime_error("view cannot infer dimension: tensor size is not divisible");
            }
            else
            {
                // The missing axis is numel divided by the product of the known axes.
                shape[static_cast<size_t>(infer_index)] = static_cast<int>(numel / known_product);
            }
        }
        else if (known_product != numel)
        {
            throw std::runtime_error("view shape is incompatible with tensor size");
        }

        return shape;
    }

#if DEEPLEARNLIB_ENABLE_CUDA


    // Block count for element-wise kernels: 256 threads per block.
    auto conversion_launch(int count) -> dim3
    {
        constexpr int kThreads = 256;
        return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
    }

    constexpr int kInplaceThreads = 256;

    // dst += src, one thread per element.
    __global__ void add_inplace_f32_kernel(float* dst, const float* src, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] += src[index];
        }
    }


    // dst *= scalar, one thread per element.
    __global__ void mul_inplace_f32_kernel(float* dst, float scalar, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] *= scalar;
        }
    }


    // out = lhs * rhs. The result goes to a separate buffer; the operands stay.
    __global__ void mul_into_f32_kernel(const float* lhs, const float* rhs, float* out, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            out[index] = lhs[index] * rhs[index];
        }
    }


    // Clamps each element to [lo, hi].
    __global__ void clamp_inplace_f32_kernel(float* dst, float lo, float hi, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            const float value = dst[index];
            dst[index] = value < lo ? lo : (value > hi ? hi : value);
        }
    }


    // dst += scale * src. A negative scale subtracts without a second kernel.
    __global__ void add_scaled_inplace_f32_kernel(float* dst, const float* src, float scale, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] += scale * src[index];
        }
    }


    // SGD: weight -= lr * clip(grad + decay * weight). clip <= 0 disables the clamp.
    __global__ void sgd_update_f32_kernel(float* weights, const float* grad, float lr, float decay, float clip,
        int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index >= count)
        {
            return;
        }
        // Decay is added into the gradient. clip <= 0 leaves that update unclamped.
        float update = grad[index] + (decay * weights[index]);
        if (clip > 0.0F)
        {
            update = update < -clip ? -clip : (update > clip ? clip : update);
        }
        weights[index] -= lr * update;
    }


    // SGD with momentum: velocity = momentum * velocity + update, then weight -= lr * velocity.
    __global__ void sgd_momentum_update_f32_kernel(float* weights, const float* grad, float* velocity, float lr,
        float momentum, float decay, float clip, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index >= count)
        {
            return;
        }
        // Same decay and optional clip as plain SGD. Velocity then accumulates the update.
        float update = grad[index] + (decay * weights[index]);
        if (clip > 0.0F)
        {
            update = update < -clip ? -clip : (update > clip ? clip : update);
        }
        const float velocity_value = (momentum * velocity[index]) + update;
        velocity[index] = velocity_value;
        weights[index] -= lr * velocity_value;
    }


    // Adds the bias to every sample: element i receives bias[i % features].
    __global__ void add_row_f32_kernel(float* dst, const float* bias, int count, int features)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            // i % features repeats the same bias on every row.
            dst[index] += bias[index % features];
        }
    }


    // Sums a column across rows and adds beta times the value already in dst.
    __global__ void add_sum_rows_f32_kernel(float* dst, const float* src, int batch, int cols, float beta)
    {
        const int col = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (col >= cols)
        {
            return;
        }
        float acc = 0.0F;
        for (int row = 0; row < batch; ++row)
        {
            acc += src[(row * cols) + col];
        }
        // beta keeps the previous destination: 0 replaces it, 1 accumulates the column sum.
        dst[col] = (beta * dst[col]) + acc;
    }


    // dst += scalar, one thread per element.
    __global__ void add_scalar_f32_kernel(float* dst, float scalar, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] += scalar;
        }
    }


    // 2D transpose into a new buffer: output[column, row] = input[row, column].
    __global__ void transpose_2d_f32_kernel(const float* input, float* output, int rows, int cols, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            const int row = index / cols;
            const int col = index % cols;
            output[(col * rows) + row] = input[index];
        }
    }


    constexpr int kReduceThreads = 256;

    // Sum of every element. Partial sum in shared memory, atomicAdd from thread 0 of the block.
    __global__ void sum_f32_kernel(const float* input, float* out, int count)
    {
        __shared__ float shared_sum[kReduceThreads];
        float partial = 0.0F;
        for (int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x); index < count;
             index += static_cast<int>(blockDim.x * gridDim.x))
        {
            partial += input[index];
        }
        shared_sum[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1)
        {
            if (static_cast<int>(threadIdx.x) < stride)
            {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(out, shared_sum[0]);
        }
    }

    // Sets the flag when any element is not finite. Any store of 1 is enough, so there is no atomic.
    __global__ void any_nonfinite_f32_kernel(const float* input, int* flag, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count && !isfinite(input[index]))
        {
            // A plain store of 1 is enough; the flag does not need an atomic.
            *flag = 1;
        }
    }
#endif

} // namespace

#if DEEPLEARNLIB_ENABLE_CUDA
namespace
{
    constexpr std::size_t kCublasWorkspaceBytes = 64ULL * 1024ULL * 1024ULL;
}

// cuBLAS handle and a 64 MiB workspace. The math mode turns on TF32 tensor cores.
CublasContext::CublasContext()
{
    CHECK_CUBLAS(cublasCreate(&handle_));
    // TF32: faster GEMM, with a shorter mantissa than full fp32.
    CHECK_CUBLAS(cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUDA(cudaMalloc(&workspace_, kCublasWorkspaceBytes));
    CHECK_CUBLAS(cublasSetWorkspace(handle_, workspace_, kCublasWorkspaceBytes));
}

// Releases the handle and the workspace. Errors during process shutdown are ignored.
CublasContext::~CublasContext()
{
    if (handle_ != nullptr)
    {
        static_cast<void>(cublasDestroy(handle_));
        handle_ = nullptr;
    }
    if (workspace_ != nullptr)
    {
        static_cast<void>(cudaFree(workspace_));
        workspace_ = nullptr;
    }
}

// One context instance per process.
auto CublasContext::handle() -> cublasHandle_t
{
    static CublasContext context;
    return context.handle_;
}

// Shorthand for CublasContext::handle, used by the layers.
auto get_cublas_handle() -> cublasHandle_t
{
    return CublasContext::handle();
}

namespace
{
    thread_local cudaStream_t g_current_stream = 0;
}

// CUDA stream of the current thread. Zero is the default stream.
auto current_stream() -> cudaStream_t
{
    return g_current_stream;
}

// Replaces the thread stream. cuBLAS is updated only by StreamGuard.
auto set_current_stream(cudaStream_t stream) -> void
{
    g_current_stream = stream;
}

// Saves the previous stream and sets the new one on cuBLAS as well.
StreamGuard::StreamGuard(cudaStream_t stream)
    : previous_(current_stream())
{
    set_current_stream(stream);
    CHECK_CUBLAS(cublasSetStream(get_cublas_handle(), stream));
}

// Restores the previous thread stream and cuBLAS stream, including when the body threw.
StreamGuard::~StreamGuard()
{
    set_current_stream(previous_);
    static_cast<void>(cublasSetStream(get_cublas_handle(), previous_));
}
#endif

// Tensor with an empty shape on the CPU, fp32. The product of an empty shape stays 1, so this is a scalar.
Tensor::Tensor()
    : Tensor(std::vector<int> {}, Device::CPU, Dtype::Float32)
{
}

// Allocates a buffer of the product of the dimensions. GPU: cudaMalloc, CPU: zeroed operator new.
// The shared pointer, with CudaDeleter or CpuDeleter, releases the memory.
Tensor::Tensor(std::vector<int> shape, Device device_type, Dtype dtype)
    : shape_(std::move(shape))
    , device_(device_type)
    , dtype_(dtype)
    , size_(calculate_size(shape_))
{
    compute_strides();
    const std::size_t bytes = nbytes();
#if DEEPLEARNLIB_ENABLE_CUDA
    if (device_ == Device::GPU)
    {
        int device_count { 0 };
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        if (device_count == 0)
        {
            throw std::runtime_error("No CUDA-capable devices found");
        }

        void* gpu_pointer { nullptr };
        // cudaMalloc does not zero this buffer.
        CHECK_CUDA(cudaMalloc(&gpu_pointer, bytes));
        data_ = std::shared_ptr<float>(static_cast<float*>(gpu_pointer), CudaDeleter());
    }
    else
    {
        void* cpu_pointer = ::operator new(bytes);
        std::memset(cpu_pointer, 0, bytes);
        data_ = std::shared_ptr<float>(static_cast<float*>(cpu_pointer), CpuDeleter());
    }
#else
    if (device_ == Device::GPU)
    {
        throw std::runtime_error("CUDA support is not enabled");
    }
    void* cpu_pointer = ::operator new(bytes);
    std::memset(cpu_pointer, 0, bytes);
    data_ = std::shared_ptr<float>(static_cast<float*>(cpu_pointer), CpuDeleter());
#endif
}

// View onto an existing buffer, with no allocation. It does not check strides; view() does that before this call.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Tensor::Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data_ptr, Device device_type,
    Dtype dtype)
    : shape_(std::move(shape))
    , strides_(std::move(strides))
    , device_(device_type)
    , dtype_(dtype)
    , size_(calculate_size(shape_))
    , data_(std::move(data_ptr))
{
}

// Metadata and pointer accessors: shape, strides, size, device, dtype, bytes, data.
// The buffer is not copied.
auto Tensor::get_shape() const -> const std::vector<int>&
{
    return shape_;
}

// One line for error messages: shape, dtype, device, and element count.
auto Tensor::describe() const -> std::string
{
    const char* device_name = (device_ == Device::GPU) ? "GPU" : "CPU";
    return format_shape(shape_) + " " + dtype_name(dtype_) + " " + device_name + " n=" + std::to_string(size_);
}

auto Tensor::get_strides() const -> const std::vector<int>&
{
    return strides_;
}

auto Tensor::get_size() const -> size_t
{
    return size_;
}

auto Tensor::get_device() const -> Device
{
    return device_;
}

auto Tensor::get_dtype() const -> Dtype
{
    return dtype_;
}

auto Tensor::element_size() const -> std::size_t
{
    return ::dl::element_size(dtype_);
}

auto Tensor::nbytes() const -> std::size_t
{
    return size_ * element_size();
}

auto Tensor::get_data() const -> const float*
{
    return data();
}

auto Tensor::data() -> float*
{
    return data_.get();
}

auto Tensor::data() const -> const float*
{
    return data_.get();
}

#if DEEPLEARNLIB_ENABLE_CUDA


// Returns a view when the dtype matches. Another type is not implemented, so this throws.
auto Tensor::to_dtype(Dtype dtype, cudaStream_t stream) const -> Tensor
{
    (void)stream;
    if (dtype != dtype_)
    {
        throw std::runtime_error("to_dtype only supports the tensor dtype");
    }
    return view(shape_);
}

#endif

// Recomputes strides as contiguous row-major from the current shape.
auto Tensor::compute_strides() -> void
{
    strides_ = make_contiguous_strides(shape_);
}

// True when the strides match row-major. A dimension of length 1 may have any stride.
auto Tensor::is_contiguous() const -> bool
{
    if (shape_.empty())
    {
        return true;
    }

    int expected_stride { 1 };
    for (int dim_idx = static_cast<int>(shape_.size()) - 1; dim_idx >= 0; --dim_idx)
    {
        // A length-1 axis may keep any stride and still count as contiguous.
        if (shape_[dim_idx] != 1 && strides_[dim_idx] != expected_stride)
        {
            return false;
        }
        expected_stride *= shape_[dim_idx];
    }
    return true;
}

// Throws when the tensor is not on the GPU, or the pointer is null while the size is non-zero.
auto Tensor::ensure_gpu(const char* op_name) const -> void
{
    if (device_ != Device::GPU)
    {
        throw std::runtime_error(std::string(op_name) + " requires a GPU tensor");
    }
    if (size_ > 0 && data_.get() == nullptr)
    {
        throw std::runtime_error(std::string(op_name) + " requires a valid device pointer");
    }
}

// Shared checks for a two-argument operation: GPU, equal size, contiguous, and the same dtype.
auto Tensor::ensure_binary_op(const Tensor& other, const char* op_name) const -> void
{
    ensure_gpu(op_name);
    other.ensure_gpu(op_name);
    if (size_ != other.size_)
    {
        throw std::runtime_error(std::string(op_name) + " requires tensors of equal size");
    }
    if (!is_contiguous() || !other.is_contiguous())
    {
        throw std::runtime_error(std::string(op_name) + " requires contiguous tensors");
    }
    if (dtype_ != other.dtype_)
    {
        throw std::runtime_error(std::string(op_name) + " requires tensors of equal dtype");
    }
}

#if DEEPLEARNLIB_ENABLE_CUDA
namespace
{

    struct GemmPlan
    {
        int M { 0 };
        int N { 0 };
        int K { 0 };
        int lda { 0 };
        int ldb { 0 };
        int ldc { 0 };
        cublasOperation_t trans_a { CUBLAS_OP_N };
        cublasOperation_t trans_b { CUBLAS_OP_N };
        std::vector<int> result_shape;
    };

    // Computes M, N, K, and the leading dimensions for a row-major matmul.
    // cuBLAS is column-major, so the transpose flags of A and B are swapped.
    auto plan_rowmajor_gemm(const Tensor& a, const Tensor& b, bool transpose_a, bool transpose_b) -> GemmPlan
    {
        if (a.get_device() != Device::GPU || b.get_device() != Device::GPU)
        {
            throw std::runtime_error("matmul requires both tensors to reside on the GPU");
        }
        if (a.data() == nullptr || b.data() == nullptr)
        {
            throw std::runtime_error("matmul requires valid device pointers");
        }
        if (a.get_shape().empty() || b.get_shape().empty())
        {
            throw std::runtime_error("matmul requires non-scalar tensors");
        }
        if (a.get_dtype() != b.get_dtype())
        {
            throw std::runtime_error("matmul requires tensors of equal dtype");
        }
        if (transpose_a && a.get_shape().size() != 2)
        {
            throw std::runtime_error("matmul transpose_a currently requires a rank-2 tensor");
        }
        if (transpose_b && b.get_shape().size() != 2)
        {
            throw std::runtime_error("matmul transpose_b currently requires a rank-2 tensor");
        }

        const std::vector<int>& a_shape = a.get_shape();
        const std::vector<int>& b_shape = b.get_shape();
        GemmPlan plan;
        plan.M = transpose_a ? a_shape[1] : static_cast<int>(a.get_size() / static_cast<size_t>(a_shape.back()));
        plan.K = transpose_a ? a_shape[0] : a_shape.back();
        const int other_k = transpose_b ? b_shape[1] : b_shape.front();
        plan.N = transpose_b ? b_shape[0] : static_cast<int>(b.get_size() / static_cast<size_t>(other_k));
        if (plan.K != other_k)
        {
            throw std::runtime_error("matmul inner dimensions must match (" + std::to_string(plan.K) + " vs "
                + std::to_string(other_k) + ")");
        }
        if (plan.K <= 0)
        {
            throw std::runtime_error("matmul inner dimension must be positive");
        }

        plan.result_shape.reserve(a_shape.size() + b_shape.size() - 2);
        if (transpose_a)
        {
            plan.result_shape.push_back(a_shape[1]);
        }
        else
        {
            plan.result_shape.insert(plan.result_shape.end(), a_shape.begin(), a_shape.end() - 1);
        }
        if (transpose_b)
        {
            plan.result_shape.push_back(b_shape[0]);
        }
        else
        {
            plan.result_shape.insert(plan.result_shape.end(), b_shape.begin() + 1, b_shape.end());
        }
        if (plan.result_shape.empty())
        {
            plan.result_shape.push_back(1);
        }

        // Row-major C = A*B is column-major C^T = B^T * A^T, so the transpose flags are swapped.
        // trans_a follows transpose_b, and trans_b follows transpose_a.
        plan.trans_a = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        plan.trans_b = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        // lda describes B and ldb describes A, because those matrices are passed to cuBLAS in that order.
        plan.lda = transpose_b ? b_shape[1] : plan.N;
        plan.ldb = a_shape.back();
        plan.ldc = plan.N;
        return plan;
    }

    // cublasGemmEx for the plan. B is passed first, and N and M are swapped, because cuBLAS is column-major.
    auto launch_rowmajor_gemm(const Tensor& a, const Tensor& b, Tensor& c, const GemmPlan& plan, float beta) -> void
    {
        if (plan.M == 0 || plan.N == 0)
        {
            return;
        }

        const float alpha { 1.0F };
        CHECK_CUBLAS(cublasSetStream(get_cublas_handle(), current_stream()));
        // B is passed first, with dimensions N, M, K. CUDA_R_32F storage, CUBLAS_COMPUTE_32F_FAST_TF32 compute (tensor-core multiply, fp32 accumulate).
        CHECK_CUBLAS(cublasGemmEx(get_cublas_handle(), plan.trans_a, plan.trans_b, plan.N, plan.M, plan.K, &alpha,
            b.data(), CUDA_R_32F, plan.lda, a.data(), CUDA_R_32F, plan.ldb, &beta, c.data(), CUDA_R_32F, plan.ldc,
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    }

} // namespace
#endif

// Matrix product on the GPU. The result is a new tensor; beta = 0 overwrites it from zero.
auto Tensor::matmul(const Tensor& other, bool transpose_a, bool transpose_b) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("matmul requires CUDA/cuBLAS support");
#else
    if (!is_contiguous() || !other.is_contiguous())
    {
        throw std::runtime_error("matmul requires contiguous row-major tensors");
    }
    const GemmPlan plan = plan_rowmajor_gemm(*this, other, transpose_a, transpose_b);
    Tensor result(plan.result_shape, Device::GPU, dtype_);
    // beta 0 overwrites the new tensor from zero.
    launch_rowmajor_gemm(*this, other, result, plan, 0.0F);
    return result;
#endif
}

// Product written into out. beta scales the current contents, so 1 accumulates and 0 overwrites.
auto Tensor::matmul_into(const Tensor& other, Tensor& out, bool transpose_a, bool transpose_b, float beta) const
    -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("matmul_into requires CUDA/cuBLAS support");
#else
    if (!is_contiguous() || !other.is_contiguous())
    {
        throw std::runtime_error("matmul_into requires contiguous row-major tensors");
    }
    const GemmPlan plan = plan_rowmajor_gemm(*this, other, transpose_a, transpose_b);
    if (out.get_device() != Device::GPU)
    {
        throw std::runtime_error("matmul_into requires a GPU output tensor");
    }
    if (out.get_dtype() != dtype_)
    {
        throw std::runtime_error("matmul_into requires the output dtype to match the operands");
    }
    if (out.get_shape() != plan.result_shape)
    {
        throw std::runtime_error("matmul_into output shape " + format_shape(out.get_shape()) + " does not match GEMM "
            + format_shape(plan.result_shape));
    }
    // beta scales what is already in out: 0 overwrites, 1 accumulates.
    launch_rowmajor_gemm(*this, other, out, plan, beta);
    return out;
#endif
}

// Returns the tensor already in the slot when shape, device, and dtype match. Otherwise it allocates.
// A stable shape must not call cudaMalloc on every training step.
auto Tensor::ensure(std::optional<Tensor>& slot, const std::vector<int>& shape, Device device, Dtype dtype) -> Tensor&
{
    // Reuse the slot when shape, device, and dtype are unchanged, so a stable shape does not cudaMalloc every step.
    if (!slot.has_value() || slot->get_shape() != shape || slot->get_device() != device || slot->get_dtype() != dtype)
    {
        slot = Tensor(shape, device, dtype);
    }
    return *slot;
}

// Element-wise sum. Copies the left-hand side and adds the right-hand side in place.
auto Tensor::operator+(const Tensor& other) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator+ requires CUDA support");
#else
    ensure_binary_op(other, "operator+");
    Tensor result(shape_, Device::GPU, dtype_);
    if (size_ == 0)
    {
        return result;
    }
    CHECK_CUDA(cudaMemcpyAsync(result.data(), data(), nbytes(), cudaMemcpyDeviceToDevice, current_stream()));
    result.add_(other);
    return result;
#endif
}

// Difference: a copy of the left-hand side plus the right-hand side scaled by -1.
auto Tensor::operator-(const Tensor& other) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator- requires CUDA support");
#else
    ensure_binary_op(other, "operator-");
    Tensor result(shape_, Device::GPU, dtype_);
    if (size_ == 0)
    {
        return result;
    }
    CHECK_CUDA(cudaMemcpyAsync(result.data(), data(), nbytes(), cudaMemcpyDeviceToDevice, current_stream()));
    // Scale -1 subtracts without a separate kernel.
    result.add_scaled_(other, -1.0F);
    return result;
#endif
}

// Element-wise product into a new tensor.
auto Tensor::operator*(const Tensor& other) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator* requires CUDA support");
#else
    ensure_binary_op(other, "operator*");
    Tensor result(shape_, Device::GPU, dtype_);
    if (size_ == 0)
    {
        return result;
    }
    mul_into(other, result);
    return result;
#endif
}

// Multiply by a scalar: copy, then mul_ in place.
auto Tensor::operator*(float scalar) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator* requires CUDA support");
#else
    ensure_gpu("operator*");
    if (!is_contiguous())
    {
        throw std::runtime_error("operator* requires a contiguous tensor");
    }

    Tensor result(shape_, Device::GPU, dtype_);
    if (size_ == 0)
    {
        return result;
    }
    CHECK_CUDA(cudaMemcpyAsync(result.data(), data(), nbytes(), cudaMemcpyDeviceToDevice, current_stream()));
    result.mul_(scalar);
    return result;
#endif
}

// Add a scalar to a copy.
auto Tensor::operator+(float scalar) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator+ requires CUDA support");
#else
    ensure_gpu("operator+");
    if (!is_contiguous())
    {
        throw std::runtime_error("operator+ requires a contiguous tensor");
    }

    Tensor result(shape_, Device::GPU, dtype_);
    if (size_ == 0)
    {
        return result;
    }
    CHECK_CUDA(cudaMemcpyAsync(result.data(), data(), nbytes(), cudaMemcpyDeviceToDevice, current_stream()));
    const int count = static_cast<int>(size_);
        add_scalar_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        result.data(), scalar, count);

    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

// this += other, in place.
auto Tensor::add_(const Tensor& other) -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("add_ requires CUDA support");
#else
    ensure_binary_op(other, "add_");
    if (size_ == 0)
    {
        return *this;
    }

    const int count = static_cast<int>(size_);
        add_inplace_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        data(), other.data(), count);

    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

// this *= scalar, in place.
auto Tensor::mul_(float scalar) -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("mul_ requires CUDA support");
#else
    ensure_gpu("mul_");
    if (!is_contiguous())
    {
        throw std::runtime_error("mul_ requires a contiguous tensor");
    }
    if (size_ == 0)
    {
        return *this;
    }

    const int count = static_cast<int>(size_);
        mul_inplace_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        data(), scalar, count);

    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

// out = this * other. out must have the same size and dtype as the operands.
auto Tensor::mul_into(const Tensor& other, Tensor& out) const -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("mul_into requires CUDA support");
#else
    ensure_binary_op(other, "mul_into");
    if (out.get_device() != Device::GPU)
    {
        throw std::runtime_error("mul_into requires a GPU output tensor");
    }
    if (out.get_dtype() != dtype_ || out.get_size() != size_)
    {
        throw std::runtime_error("mul_into output must match operand shape and dtype");
    }
    if (size_ == 0)
    {
        return out;
    }

    const int count = static_cast<int>(size_);
        mul_into_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        data(), other.data(), out.data(), count);

    CHECK_CUDA(cudaGetLastError());
    return out;
#endif
}

// this += scale * other, in place.
auto Tensor::add_scaled_(const Tensor& other, float scale) -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("add_scaled_ requires CUDA support");
#else
    ensure_binary_op(other, "add_scaled_");
    if (size_ == 0)
    {
        return *this;
    }

    const int count = static_cast<int>(size_);
        add_scaled_inplace_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        data(), other.data(), scale, count);

    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

// One SGD step on the weights stored in this tensor.
auto Tensor::sgd_update_(const Tensor& grad, float lr, float decay, float clip) -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("sgd_update_ requires CUDA support");
#else
    ensure_binary_op(grad, "sgd_update_");
    if (size_ == 0)
    {
        return *this;
    }

    const int count = static_cast<int>(size_);
        sgd_update_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        data(), grad.data(), lr, decay, clip, count);

    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

// SGD step with a velocity buffer of the same shape as the weights.
auto Tensor::sgd_momentum_update_(const Tensor& grad, Tensor& velocity, float lr, float momentum, float decay,
    float clip) -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("sgd_momentum_update_ requires CUDA support");
#else
    ensure_binary_op(grad, "sgd_momentum_update_");
    ensure_binary_op(velocity, "sgd_momentum_update_");
    if (size_ == 0)
    {
        return *this;
    }

    const int count = static_cast<int>(size_);
        sgd_momentum_update_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        data(), grad.data(), velocity.data(), lr, momentum, decay, clip, count);

    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

// Adds a bias vector to every row of a [batch, features] matrix.
auto Tensor::add_row_(const Tensor& bias) -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("add_row_ requires CUDA support");
#else
    ensure_gpu("add_row_");
    bias.ensure_gpu("add_row_");
    if (shape_.size() != 2)
    {
        throw std::runtime_error("add_row_ requires a rank-2 [batch, features] tensor");
    }
    if (!is_contiguous() || !bias.is_contiguous())
    {
        throw std::runtime_error("add_row_ requires contiguous tensors");
    }
    if (dtype_ != bias.dtype_)
    {
        throw std::runtime_error("add_row_ requires tensors of equal dtype");
    }
    const int features = shape_[1];
    if (static_cast<int>(bias.get_size()) != features)
    {
        throw std::runtime_error("add_row_ bias size must match the feature dimension");
    }
    if (size_ == 0)
    {
        return *this;
    }

    const int count = static_cast<int>(size_);
        add_row_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        data(), bias.data(), count, features);

    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

// this[j] = beta * this[j] + the sum of the rows of column j of the source matrix.
auto Tensor::add_sum_rows_(const Tensor& matrix, float beta) -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("add_sum_rows_ requires CUDA support");
#else
    ensure_gpu("add_sum_rows_");
    matrix.ensure_gpu("add_sum_rows_");
    if (matrix.get_shape().size() != 2)
    {
        throw std::runtime_error("add_sum_rows_ requires a rank-2 [batch, features] source");
    }
    if (!is_contiguous() || !matrix.is_contiguous())
    {
        throw std::runtime_error("add_sum_rows_ requires contiguous tensors");
    }
    if (dtype_ != matrix.get_dtype())
    {
        throw std::runtime_error("add_sum_rows_ requires tensors of equal dtype");
    }
    const int cols = matrix.get_shape()[1];
    const int batch = matrix.get_shape()[0];
    if (static_cast<int>(size_) != cols)
    {
        throw std::runtime_error("add_sum_rows_ destination size must match the source feature dimension");
    }
    if (size_ == 0)
    {
        return *this;
    }

        add_sum_rows_f32_kernel<<<conversion_launch(cols), kInplaceThreads, 0, current_stream()>>>(
        data(), matrix.data(), batch, cols, beta);

    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

// A copy clamped to [lo, hi].
auto Tensor::clamp(float lo, float hi) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("clamp requires CUDA support");
#else
    ensure_gpu("clamp");
    if (!is_contiguous())
    {
        throw std::runtime_error("clamp requires a contiguous tensor");
    }
    if (lo > hi)
    {
        throw std::runtime_error("clamp requires lo <= hi");
    }

    Tensor result(shape_, Device::GPU, dtype_);
    if (size_ == 0)
    {
        return result;
    }
    CHECK_CUDA(cudaMemcpyAsync(result.data(), data(), nbytes(), cudaMemcpyDeviceToDevice, current_stream()));
    result.clamp_(lo, hi);
    return result;
#endif
}

// Clamp to [lo, hi] in place.
auto Tensor::clamp_(float lo, float hi) -> Tensor&
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("clamp_ requires CUDA support");
#else
    ensure_gpu("clamp_");
    if (!is_contiguous())
    {
        throw std::runtime_error("clamp_ requires a contiguous tensor");
    }
    if (lo > hi)
    {
        throw std::runtime_error("clamp_ requires lo <= hi");
    }
    if (size_ == 0)
    {
        return *this;
    }

    const int count = static_cast<int>(size_);
        clamp_inplace_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        data(), lo, hi, count);

    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

// Looks for NaN or Inf. On the GPU the flag comes back to the host, so the stream is synchronized.
auto Tensor::has_non_finite() const -> bool
{
    if (size_ == 0)
    {
        return false;
    }
#if DEEPLEARNLIB_ENABLE_CUDA
    if (device_ == Device::GPU)
    {
        const Tensor* source = this;
        // One device flag for the process, cleared on the current stream before each launch.
        static int* flag = nullptr;
        if (flag == nullptr)
        {
            CHECK_CUDA(cudaMalloc(&flag, sizeof(int)));
        }
        CHECK_CUDA(cudaMemsetAsync(flag, 0, sizeof(int), current_stream()));
        const int count = static_cast<int>(source->get_size());
        any_nonfinite_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
            source->data(), flag, count);
        CHECK_CUDA(cudaGetLastError());
        int host_flag = 0;
        CHECK_CUDA(cudaMemcpyAsync(&host_flag, flag, sizeof(int), cudaMemcpyDeviceToHost, current_stream()));
        // The flag lives on the device; the host read waits for the kernel and the memcpy.
        CHECK_CUDA(cudaStreamSynchronize(current_stream()));
        return host_flag != 0;
    }
#endif
    const float* host = get_data();
    for (size_t index = 0; index < size_; ++index)
    {
        if (!std::isfinite(host[index]))
        {
            return true;
        }
    }
    return false;
}

// Under DEBUG_NUMERICS, throws when has_non_finite is true. Otherwise it compiles to nothing.
auto Tensor::assert_finite(const char* context) const -> void
{
#ifdef DEBUG_NUMERICS
    if (has_non_finite())
    {
        const std::string where = context == nullptr ? "Tensor" : context;
        throw std::runtime_error("NaN detected in " + where);
    }
#else
    (void)context;
#endif
}

// Sum of every element into a GPU scalar. An axis other than -1 is not implemented.
auto Tensor::sum(int dim) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("sum requires CUDA support");
#else
    ensure_gpu("sum");
    if (dim != -1)
    {
        throw std::runtime_error("sum along a specific axis is not implemented; use dim = -1");
    }
    if (!is_contiguous())
    {
        throw std::runtime_error("sum requires a contiguous tensor");
    }

    Tensor result({ 1 }, Device::GPU, Dtype::Float32);
    CHECK_CUDA(cudaMemsetAsync(result.data(), 0, sizeof(float), current_stream()));
    if (size_ > 0)
    {
        const Tensor* source = this;
        const int count = static_cast<int>(source->get_size());
        const int blocks = std::max(1, (count + kReduceThreads - 1) / kReduceThreads);
        // The kernel grid-strides, so capping the launch at 1024 blocks still covers every element.
        sum_f32_kernel<<<static_cast<unsigned int>(std::min(blocks, 1024)), kReduceThreads, 0, current_stream()>>>(
            source->data(), result.data(), count);
        CHECK_CUDA(cudaGetLastError());
    }
    return result;
#endif
}

// A new view of the same buffer. Requires a contiguous layout and a matching element count.
auto Tensor::view(const std::vector<int>& new_shape) const -> Tensor
{
    if (!is_contiguous())
    {
        throw std::runtime_error("view requires a contiguous tensor");
    }

    std::vector<int> shape = infer_view_shape(new_shape, size_);
    std::vector<int> strides = make_contiguous_strides(shape);
    return Tensor(std::move(shape), std::move(strides), data_, device_, dtype_);
}

// A view with the same shape. Returns a tensor without handing off exclusive ownership of the buffer.
auto Tensor::as_view() const -> Tensor
{
    return view(shape_);
}

// Transpose of a 2D matrix into a new buffer.
auto Tensor::transpose() const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("transpose requires CUDA support");
#else
    ensure_gpu("transpose");
    if (shape_.size() != 2)
    {
        throw std::runtime_error("transpose currently supports 2D tensors only");
    }
    if (!is_contiguous())
    {
        throw std::runtime_error("transpose requires a contiguous tensor");
    }

    const int rows { shape_[0] };
    const int cols { shape_[1] };
    Tensor result({ cols, rows }, Device::GPU, dtype_);
    if (size_ == 0 || rows == 0 || cols == 0)
    {
        return result;
    }

    const int count = static_cast<int>(size_);
        transpose_2d_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
        data(), result.data(), rows, cols, count);

    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

// A tensor of the same shape, filled with zeros. On the GPU the memset runs on the current stream.
auto Tensor::zeros_like(const Tensor& other) -> Tensor
{
    Tensor result(other.shape_, other.device_, other.dtype_);
#if DEEPLEARNLIB_ENABLE_CUDA
    if (result.device_ == Device::GPU && result.size_ > 0)
    {
        // cudaMalloc does not clear memory, so zero the GPU buffer on the current stream.
        CHECK_CUDA(cudaMemsetAsync(result.data(), 0, result.nbytes(), current_stream()));
        CHECK_CUDA(cudaGetLastError());
    }
#endif
    return result;
}

#if DEEPLEARNLIB_ENABLE_CUDA
namespace
{

    constexpr int kPinnedSlots = 4;

    struct PinnedSlot
    {
        float* ptr { nullptr };
        size_t bytes { 0 };
        cudaEvent_t event { nullptr };
        bool recorded { false };
    };

    struct PinnedPool
    {
        PinnedSlot slots[kPinnedSlots];
        int next { 0 };

        // Frees the events and the pinned memory. Errors during destruction are ignored.
        ~PinnedPool()
        {
            for (PinnedSlot& slot : slots)
            {
                if (slot.event != nullptr)
                {
                    static_cast<void>(cudaEventDestroy(slot.event));
                }
                if (slot.ptr != nullptr)
                {
                    static_cast<void>(cudaFreeHost(slot.ptr));
                }
            }
        }

        // The next of the four slots. Waits until the previous memcpy has finished reading the buffer, and grows the slot when needed.
        auto acquire(size_t bytes) -> float*
        {
            PinnedSlot& slot = slots[next];
            next = (next + 1) % kPinnedSlots;
            if (slot.recorded)
            {
                // The host waits until the previous memcpy has finished reading this slot.
                CHECK_CUDA(cudaEventSynchronize(slot.event));
                slot.recorded = false;
            }
            if (slot.bytes < bytes)
            {
                if (slot.ptr != nullptr)
                {
                    CHECK_CUDA(cudaFreeHost(slot.ptr));
                    slot.ptr = nullptr;
                }
                CHECK_CUDA(cudaMallocHost(&slot.ptr, bytes));
                slot.bytes = bytes;
            }
            if (slot.event == nullptr)
            {
                CHECK_CUDA(cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming));
            }
            return slot.ptr;
        }

        // Records an event on the stream so acquire does not overwrite the slot before the memcpy finishes.
        auto record(float* ptr, cudaStream_t stream) -> void
        {
            for (PinnedSlot& slot : slots)
            {
                if (slot.ptr == ptr)
                {
                    CHECK_CUDA(cudaEventRecord(slot.event, stream));
                    slot.recorded = true;
                    return;
                }
            }
        }
    };

    // One pinned pool per process. Four slots let successive H2D and D2H copies overlap.
    auto pinned_pool() -> PinnedPool&
    {
        static PinnedPool pool;
        return pool;
    }

} // namespace
#endif

// Host copy. From the GPU it goes through the pinned pool; the synchronize waits until that memcpy finishes.
auto Tensor::to_host(cudaStream_t stream) const -> std::vector<float>
{
        std::vector<float> host(size_);
    if (size_ == 0)
    {
        return host;
    }
    if (data_.get() == nullptr)
    {
        throw std::runtime_error("to_host requires a valid data pointer");
    }
#if DEEPLEARNLIB_ENABLE_CUDA
    if (device_ == Device::GPU)
    {
        const size_t bytes = size_ * sizeof(float);
        // Stage the D2H copy in the pinned pool. The synchronize below publishes a host float vector.
        float* pinned = pinned_pool().acquire(bytes);
        CHECK_CUDA(cudaMemcpyAsync(pinned, data_.get(), bytes, cudaMemcpyDeviceToHost, stream));
        pinned_pool().record(pinned, stream);
        // to_host synchronizes here because the caller needs a host float vector, and the D2H copy lands in the pinned slot first.
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::memcpy(host.data(), pinned, bytes);
        return host;
    }
#endif
    std::copy(data_.get(), data_.get() + static_cast<std::ptrdiff_t>(size_), host.begin());
    return host;
}

// Checks that the buffer has as many elements as the product of the shape, then calls the pointer overload.
auto Tensor::from_host(const std::vector<int>& shape, const std::vector<float>& host_data, Device device,
    cudaStream_t stream, Dtype dtype) -> Tensor
{
    // An empty shape skips the loop, so the expected count stays 1.
    size_t expected { 1 };
    for (int dimension : shape)
    {
        expected *= static_cast<size_t>(dimension);
    }
    if (expected != host_data.size())
    {
        throw std::runtime_error("from_host: host buffer size does not match the requested shape");
    }
    return from_host(shape, host_data.data(), device, stream, dtype);
}

// Creates an fp32 tensor and copies host_data. On the GPU the transfer is asynchronous, and the pool event protects the pinned slot.
auto Tensor::from_host(const std::vector<int>& shape, const float* host_data, Device device, cudaStream_t stream,
    Dtype dtype) -> Tensor
{
    Tensor result(shape, device, Dtype::Float32);
    if (result.size_ == 0)
    {
        if (dtype == Dtype::Float32)
        {
            return result;
        }
        // to_dtype accepts only the tensor's current type, which this empty tensor just set to fp32.
        return result.to_dtype(dtype, stream);
    }
    if (host_data == nullptr)
    {
        throw std::runtime_error("from_host requires a non-null host pointer");
    }
#if DEEPLEARNLIB_ENABLE_CUDA
    if (device == Device::GPU)
    {
        const size_t bytes = result.size_ * sizeof(float);
        float* pinned = pinned_pool().acquire(bytes);
        std::memcpy(pinned, host_data, bytes);
        // Async H2D through the pinned pool. record keeps the slot until the copy finishes; the host is not synchronized here.
        CHECK_CUDA(cudaMemcpyAsync(result.data(), pinned, bytes, cudaMemcpyHostToDevice, stream));
        pinned_pool().record(pinned, stream);
                return result;
    }
#endif
    (void)stream;
    std::copy(host_data, host_data + static_cast<std::ptrdiff_t>(result.size_), result.data());
        return result;
}

} // namespace dl
