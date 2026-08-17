#include "DeepLearnLib/Tensor.hpp"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#if DEEPLEARNLIB_ENABLE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#endif

namespace dl
{
namespace
{

#if DEEPLEARNLIB_ENABLE_CUDA
    struct ClampValue
    {
        float lo;
        float hi;

        __host__ __device__ auto operator()(float value) const -> float
        {
            if (value < lo)
            {
                return lo;
            }
            if (value > hi)
            {
                return hi;
            }
            return value;
        }
    };
#endif

    static auto calculate_size(const std::vector<int>& shape) -> int
    {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    }

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
    struct Transpose2D
    {
        const float* input;
        float* output;
        int rows;
        int cols;

        __host__ __device__ void operator()(int index) const
        {
            const int row = index / cols;
            const int col = index % cols;
            output[(col * rows) + row] = input[index];
        }
    };

    struct Transpose2DHalf
    {
        const __half* input;
        __half* output;
        int rows;
        int cols;

        __host__ __device__ void operator()(int index) const
        {
            const int row = index / cols;
            const int col = index % cols;
            output[(col * rows) + row] = input[index];
        }
    };

    template <typename FloatOp>
    struct HalfBinaryAdaptor
    {
        FloatOp op;

        __device__ auto operator()(__half lhs, __half rhs) const -> __half
        {
            return __float2half(op(__half2float(lhs), __half2float(rhs)));
        }
    };

    template <typename FloatOp>
    struct HalfUnaryAdaptor
    {
        FloatOp op;

        __device__ auto operator()(__half value) const -> __half
        {
            return __float2half(op(__half2float(value)));
        }
    };

    struct ScaleValue
    {
        float scale;

        __host__ __device__ auto operator()(float value) const -> float
        {
            return value * scale;
        }
    };

    struct AddScalar
    {
        float scalar;

        __host__ __device__ auto operator()(float value) const -> float
        {
            return value + scalar;
        }
    };

    __global__ void f32_to_f16_kernel(const float* input, __half* output, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            output[index] = __float2half(input[index]);
        }
    }

    __global__ void f16_to_f32_kernel(const __half* input, float* output, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            output[index] = __half2float(input[index]);
        }
    }

    auto conversion_launch(int count) -> dim3
    {
        constexpr int kThreads = 256;
        return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
    }

    constexpr int kInplaceThreads = 256;

    __global__ void add_inplace_f32_kernel(float* dst, const float* src, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] += src[index];
        }
    }

    __global__ void add_inplace_f16_kernel(__half* dst, const __half* src, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] = __float2half(__half2float(dst[index]) + __half2float(src[index]));
        }
    }

    __global__ void mul_inplace_f32_kernel(float* dst, float scalar, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] *= scalar;
        }
    }

    __global__ void mul_inplace_f16_kernel(__half* dst, float scalar, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] = __float2half(__half2float(dst[index]) * scalar);
        }
    }

    __global__ void add_scaled_inplace_f32_kernel(float* dst, const float* src, float scale, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] += scale * src[index];
        }
    }

    __global__ void add_scaled_inplace_f16_kernel(__half* dst, const __half* src, float scale, int count)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] = __float2half(__half2float(dst[index]) + (scale * __half2float(src[index])));
        }
    }

    __global__ void add_row_f32_kernel(float* dst, const float* bias, int count, int features)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] += bias[index % features];
        }
    }

    __global__ void add_row_f16_kernel(__half* dst, const __half* bias, int count, int features)
    {
        const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (index < count)
        {
            dst[index] = __float2half(__half2float(dst[index]) + __half2float(bias[index % features]));
        }
    }

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
        dst[col] = (beta * dst[col]) + acc;
    }

    __global__ void add_sum_rows_f16_kernel(__half* dst, const __half* src, int batch, int cols, float beta)
    {
        const int col = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
        if (col >= cols)
        {
            return;
        }
        float acc = 0.0F;
        for (int row = 0; row < batch; ++row)
        {
            acc += __half2float(src[(row * cols) + col]);
        }
        dst[col] = __float2half((beta * __half2float(dst[col])) + acc);
    }
#endif

} // namespace

#if DEEPLEARNLIB_ENABLE_CUDA
CublasContext::CublasContext()
{
    CHECK_CUBLAS(cublasCreate(&handle_));
    CHECK_CUBLAS(cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
}

CublasContext::~CublasContext()
{
    if (handle_ != nullptr)
    {
        static_cast<void>(cublasDestroy(handle_));
        handle_ = nullptr;
    }
}

auto CublasContext::handle() -> cublasHandle_t
{
    static CublasContext context;
    return context.handle_;
}

auto get_cublas_handle() -> cublasHandle_t
{
    return CublasContext::handle();
}

namespace
{
thread_local cudaStream_t g_current_stream = 0;
}

auto current_stream() -> cudaStream_t
{
    return g_current_stream;
}

auto set_current_stream(cudaStream_t stream) -> void
{
    g_current_stream = stream;
}

StreamGuard::StreamGuard(cudaStream_t stream)
    : previous_(current_stream())
{
    set_current_stream(stream);
    CHECK_CUBLAS(cublasSetStream(get_cublas_handle(), stream));
}

StreamGuard::~StreamGuard()
{
    set_current_stream(previous_);
    static_cast<void>(cublasSetStream(get_cublas_handle(), previous_));
}
#endif

Tensor::Tensor()
    : Tensor(std::vector<int> {}, Device::CPU, Dtype::Float32)
{
}

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

auto Tensor::get_shape() const -> const std::vector<int>&
{
    return shape_;
}

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
auto Tensor::half_data() -> __half*
{
    if (dtype_ != Dtype::Float16)
    {
        throw std::runtime_error("half_data requires a Float16 tensor");
    }
    return reinterpret_cast<__half*>(data_.get());
}

auto Tensor::half_data() const -> const __half*
{
    if (dtype_ != Dtype::Float16)
    {
        throw std::runtime_error("half_data requires a Float16 tensor");
    }
    return reinterpret_cast<const __half*>(data_.get());
}

auto Tensor::to_dtype(Dtype dtype, cudaStream_t stream) const -> Tensor
{
    if (dtype == dtype_)
    {
        return view(shape_);
    }
    Tensor result(shape_, device_, dtype);
    if (size_ == 0)
    {
        return result;
    }
    if (device_ != Device::GPU)
    {
        if (dtype_ == Dtype::Float32 && dtype == Dtype::Float16)
        {
            const float* input = data();
            auto* output = result.half_data();
            for (std::size_t index = 0; index < size_; ++index)
            {
                output[index] = __float2half(input[index]);
            }
            return result;
        }
        if (dtype_ == Dtype::Float16 && dtype == Dtype::Float32)
        {
            const __half* input = half_data();
            float* output = result.data();
            for (std::size_t index = 0; index < size_; ++index)
            {
                output[index] = __half2float(input[index]);
            }
            return result;
        }
        throw std::runtime_error("to_dtype: unsupported host conversion");
    }

    constexpr int kThreads = 256;
    const dim3 grid = conversion_launch(static_cast<int>(size_));
    if (dtype_ == Dtype::Float32 && dtype == Dtype::Float16)
    {
        f32_to_f16_kernel<<<grid, kThreads, 0, stream>>>(data(), result.half_data(), static_cast<int>(size_));
    }
    else if (dtype_ == Dtype::Float16 && dtype == Dtype::Float32)
    {
        f16_to_f32_kernel<<<grid, kThreads, 0, stream>>>(half_data(), result.data(), static_cast<int>(size_));
    }
    else
    {
        throw std::runtime_error("to_dtype: unsupported conversion");
    }
    CHECK_CUDA(cudaGetLastError());
    return result;
}
#endif

auto Tensor::compute_strides() -> void
{
    strides_ = make_contiguous_strides(shape_);
}

auto Tensor::is_contiguous() const -> bool
{
    if (shape_.empty())
    {
        return true;
    }

    int expected_stride { 1 };
    for (int dim_idx = static_cast<int>(shape_.size()) - 1; dim_idx >= 0; --dim_idx)
    {
        if (shape_[dim_idx] != 1 && strides_[dim_idx] != expected_stride)
        {
            return false;
        }
        expected_stride *= shape_[dim_idx];
    }
    return true;
}

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

    plan.trans_a = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    plan.trans_b = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    plan.lda = transpose_b ? b_shape[1] : plan.N;
    plan.ldb = a_shape.back();
    plan.ldc = plan.N;
    return plan;
}

auto launch_rowmajor_gemm(const Tensor& a, const Tensor& b, Tensor& c, const GemmPlan& plan, float beta) -> void
{
    if (plan.M == 0 || plan.N == 0)
    {
        return;
    }

    const float alpha { 1.0F };
    CHECK_CUBLAS(cublasSetStream(get_cublas_handle(), current_stream()));
    if (a.get_dtype() == Dtype::Float16)
    {
        CHECK_CUBLAS(cublasGemmEx(get_cublas_handle(), plan.trans_a, plan.trans_b, plan.N, plan.M, plan.K, &alpha,
            b.data(), CUDA_R_16F, plan.lda, a.data(), CUDA_R_16F, plan.ldb, &beta, c.data(), CUDA_R_16F, plan.ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    else
    {
        CHECK_CUBLAS(cublasGemmEx(get_cublas_handle(), plan.trans_a, plan.trans_b, plan.N, plan.M, plan.K, &alpha,
            b.data(), CUDA_R_32F, plan.lda, a.data(), CUDA_R_32F, plan.ldb, &beta, c.data(), CUDA_R_32F, plan.ldc,
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

} // namespace
#endif

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
    launch_rowmajor_gemm(*this, other, result, plan, 0.0F);
    return result;
#endif
}

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
    launch_rowmajor_gemm(*this, other, out, plan, beta);
    return out;
#endif
}

auto Tensor::ensure(std::optional<Tensor>& slot, const std::vector<int>& shape, Device device, Dtype dtype) -> Tensor&
{
    if (!slot.has_value() || slot->get_shape() != shape || slot->get_device() != device || slot->get_dtype() != dtype)
    {
        slot = Tensor(shape, device, dtype);
    }
    return *slot;
}

auto Tensor::operator+(const Tensor& other) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator+ requires CUDA/Thrust support");
#else
    ensure_binary_op(other, "operator+");
    Tensor result(shape_, Device::GPU, dtype_);
    if (size_ == 0)
    {
        return result;
    }

    if (dtype_ == Dtype::Float16)
    {
        auto lhs = thrust::device_pointer_cast(half_data());
        auto rhs = thrust::device_pointer_cast(other.half_data());
        auto out = thrust::device_pointer_cast(result.half_data());
        thrust::transform(thrust::cuda::par.on(current_stream()), lhs, lhs + static_cast<std::ptrdiff_t>(size_), rhs, out,
            HalfBinaryAdaptor<thrust::plus<float>> { thrust::plus<float>() });
    }
    else
    {
        auto lhs = thrust::device_pointer_cast(data());
        auto rhs = thrust::device_pointer_cast(other.data());
        auto out = thrust::device_pointer_cast(result.data());
        thrust::transform(thrust::cuda::par.on(current_stream()), lhs, lhs + static_cast<std::ptrdiff_t>(size_), rhs, out,
            thrust::plus<float>());
    }
    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

auto Tensor::operator-(const Tensor& other) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator- requires CUDA/Thrust support");
#else
    ensure_binary_op(other, "operator-");
    Tensor result(shape_, Device::GPU, dtype_);
    if (size_ == 0)
    {
        return result;
    }

    if (dtype_ == Dtype::Float16)
    {
        auto lhs = thrust::device_pointer_cast(half_data());
        auto rhs = thrust::device_pointer_cast(other.half_data());
        auto out = thrust::device_pointer_cast(result.half_data());
        thrust::transform(thrust::cuda::par.on(current_stream()), lhs, lhs + static_cast<std::ptrdiff_t>(size_), rhs, out,
            HalfBinaryAdaptor<thrust::minus<float>> { thrust::minus<float>() });
    }
    else
    {
        auto lhs = thrust::device_pointer_cast(data());
        auto rhs = thrust::device_pointer_cast(other.data());
        auto out = thrust::device_pointer_cast(result.data());
        thrust::transform(thrust::cuda::par.on(current_stream()), lhs, lhs + static_cast<std::ptrdiff_t>(size_), rhs, out,
            thrust::minus<float>());
    }
    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

auto Tensor::operator*(const Tensor& other) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator* requires CUDA/Thrust support");
#else
    ensure_binary_op(other, "operator*");
    Tensor result(shape_, Device::GPU, dtype_);
    if (size_ == 0)
    {
        return result;
    }

    if (dtype_ == Dtype::Float16)
    {
        auto lhs = thrust::device_pointer_cast(half_data());
        auto rhs = thrust::device_pointer_cast(other.half_data());
        auto out = thrust::device_pointer_cast(result.half_data());
        thrust::transform(thrust::cuda::par.on(current_stream()), lhs, lhs + static_cast<std::ptrdiff_t>(size_), rhs, out,
            HalfBinaryAdaptor<thrust::multiplies<float>> { thrust::multiplies<float>() });
    }
    else
    {
        auto lhs = thrust::device_pointer_cast(data());
        auto rhs = thrust::device_pointer_cast(other.data());
        auto out = thrust::device_pointer_cast(result.data());
        thrust::transform(thrust::cuda::par.on(current_stream()), lhs, lhs + static_cast<std::ptrdiff_t>(size_), rhs, out,
            thrust::multiplies<float>());
    }
    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

auto Tensor::operator*(float scalar) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator* requires CUDA/Thrust support");
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

    if (dtype_ == Dtype::Float16)
    {
        auto in = thrust::device_pointer_cast(half_data());
        auto out = thrust::device_pointer_cast(result.half_data());
        thrust::transform(thrust::cuda::par.on(current_stream()), in, in + static_cast<std::ptrdiff_t>(size_), out,
            HalfUnaryAdaptor<ScaleValue> { ScaleValue { scalar } });
    }
    else
    {
        auto in = thrust::device_pointer_cast(data());
        auto out = thrust::device_pointer_cast(result.data());
        thrust::transform(thrust::cuda::par.on(current_stream()), in, in + static_cast<std::ptrdiff_t>(size_), out,
            thrust::placeholders::_1 * scalar);
    }
    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

auto Tensor::operator+(float scalar) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator+ requires CUDA/Thrust support");
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

    if (dtype_ == Dtype::Float16)
    {
        auto in = thrust::device_pointer_cast(half_data());
        auto out = thrust::device_pointer_cast(result.half_data());
        thrust::transform(thrust::cuda::par.on(current_stream()), in, in + static_cast<std::ptrdiff_t>(size_), out,
            HalfUnaryAdaptor<AddScalar> { AddScalar { scalar } });
    }
    else
    {
        auto in = thrust::device_pointer_cast(data());
        auto out = thrust::device_pointer_cast(result.data());
        thrust::transform(thrust::cuda::par.on(current_stream()), in, in + static_cast<std::ptrdiff_t>(size_), out,
            thrust::placeholders::_1 + scalar);
    }
    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

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
    if (dtype_ == Dtype::Float16)
    {
        add_inplace_f16_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
            half_data(), other.half_data(), count);
    }
    else
    {
        add_inplace_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
            data(), other.data(), count);
    }
    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

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
    if (dtype_ == Dtype::Float16)
    {
        mul_inplace_f16_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
            half_data(), scalar, count);
    }
    else
    {
        mul_inplace_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
            data(), scalar, count);
    }
    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

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
    if (dtype_ == Dtype::Float16)
    {
        add_scaled_inplace_f16_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
            half_data(), other.half_data(), scale, count);
    }
    else
    {
        add_scaled_inplace_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
            data(), other.data(), scale, count);
    }
    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

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
    if (dtype_ == Dtype::Float16)
    {
        add_row_f16_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
            half_data(), bias.half_data(), count, features);
    }
    else
    {
        add_row_f32_kernel<<<conversion_launch(count), kInplaceThreads, 0, current_stream()>>>(
            data(), bias.data(), count, features);
    }
    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

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

    if (dtype_ == Dtype::Float16)
    {
        add_sum_rows_f16_kernel<<<conversion_launch(cols), kInplaceThreads, 0, current_stream()>>>(
            half_data(), matrix.half_data(), batch, cols, beta);
    }
    else
    {
        add_sum_rows_f32_kernel<<<conversion_launch(cols), kInplaceThreads, 0, current_stream()>>>(
            data(), matrix.data(), batch, cols, beta);
    }
    CHECK_CUDA(cudaGetLastError());
    return *this;
#endif
}

auto Tensor::clamp(float lo, float hi) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("clamp requires CUDA/Thrust support");
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

    if (dtype_ == Dtype::Float16)
    {
        auto in = thrust::device_pointer_cast(half_data());
        auto out = thrust::device_pointer_cast(result.half_data());
        thrust::transform(thrust::cuda::par.on(current_stream()), in, in + static_cast<std::ptrdiff_t>(size_), out,
            HalfUnaryAdaptor<ClampValue> { ClampValue { lo, hi } });
    }
    else
    {
        auto in = thrust::device_pointer_cast(data());
        auto out = thrust::device_pointer_cast(result.data());
        thrust::transform(thrust::cuda::par.on(current_stream()), in, in + static_cast<std::ptrdiff_t>(size_), out,
            ClampValue { lo, hi });
    }
    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

#if DEEPLEARNLIB_ENABLE_CUDA
namespace
{

struct IsNonFinite
{
    __host__ __device__ auto operator()(float value) const -> bool
    {
        return !isfinite(value);
    }
};

} // namespace
#endif

auto Tensor::has_non_finite() const -> bool
{
    if (size_ == 0)
    {
        return false;
    }
#if DEEPLEARNLIB_ENABLE_CUDA
    if (device_ == Device::GPU)
    {
        if (dtype_ == Dtype::Float16)
        {
            Tensor as_float = to_dtype(Dtype::Float32, current_stream());
            auto begin = thrust::device_pointer_cast(as_float.data());
            const bool found = thrust::any_of(thrust::cuda::par.on(current_stream()), begin,
                begin + static_cast<std::ptrdiff_t>(size_), IsNonFinite {});
            CHECK_CUDA(cudaGetLastError());
            return found;
        }
        auto begin = thrust::device_pointer_cast(data());
        const bool found = thrust::any_of(thrust::cuda::par.on(current_stream()), begin,
            begin + static_cast<std::ptrdiff_t>(size_), IsNonFinite {});
        CHECK_CUDA(cudaGetLastError());
        return found;
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

auto Tensor::sum(int dim) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("sum requires CUDA/Thrust support");
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
    float total { 0.0F };
    if (size_ > 0)
    {
        if (dtype_ == Dtype::Float16)
        {
            Tensor as_float = to_dtype(Dtype::Float32, current_stream());
            auto begin = thrust::device_pointer_cast(as_float.data());
            total = thrust::reduce(thrust::cuda::par.on(current_stream()), begin,
                begin + static_cast<std::ptrdiff_t>(size_), 0.0F, thrust::plus<float>());
        }
        else
        {
            auto begin = thrust::device_pointer_cast(data());
            total = thrust::reduce(thrust::cuda::par.on(current_stream()), begin,
                begin + static_cast<std::ptrdiff_t>(size_), 0.0F, thrust::plus<float>());
        }
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaMemcpyAsync(result.data(), &total, sizeof(float), cudaMemcpyHostToDevice, current_stream()));
    CHECK_CUDA(cudaStreamSynchronize(current_stream()));
    return result;
#endif
}

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

auto Tensor::transpose() const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("transpose requires CUDA/Thrust support");
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

    if (dtype_ == Dtype::Float16)
    {
        thrust::for_each(thrust::cuda::par.on(current_stream()), thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(static_cast<int>(size_)),
            Transpose2DHalf { half_data(), result.half_data(), rows, cols });
    }
    else
    {
        thrust::for_each(thrust::cuda::par.on(current_stream()), thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(static_cast<int>(size_)),
            Transpose2D { data(), result.data(), rows, cols });
    }
    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

auto Tensor::zeros_like(const Tensor& other) -> Tensor
{
    Tensor result(other.shape_, other.device_, other.dtype_);
#if DEEPLEARNLIB_ENABLE_CUDA
    if (result.device_ == Device::GPU && result.size_ > 0)
    {
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

    auto acquire(size_t bytes) -> float*
    {
        PinnedSlot& slot = slots[next];
        next = (next + 1) % kPinnedSlots;
        if (slot.recorded)
        {
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

auto pinned_pool() -> PinnedPool&
{
    static PinnedPool pool;
    return pool;
}

} // namespace
#endif

auto Tensor::to_host(cudaStream_t stream) const -> std::vector<float>
{
    if (dtype_ == Dtype::Float16)
    {
        return to_dtype(Dtype::Float32, stream).to_host(stream);
    }
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
        float* pinned = pinned_pool().acquire(bytes);
        CHECK_CUDA(cudaMemcpyAsync(pinned, data_.get(), bytes, cudaMemcpyDeviceToHost, stream));
        pinned_pool().record(pinned, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::memcpy(host.data(), pinned, bytes);
        return host;
    }
#endif
    std::copy(data_.get(), data_.get() + static_cast<std::ptrdiff_t>(size_), host.begin());
    return host;
}

auto Tensor::from_host(const std::vector<int>& shape, const std::vector<float>& host_data, Device device,
    cudaStream_t stream, Dtype dtype) -> Tensor
{
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
        CHECK_CUDA(cudaMemcpyAsync(result.data(), pinned, bytes, cudaMemcpyHostToDevice, stream));
        pinned_pool().record(pinned, stream);
        if (dtype == Dtype::Float16)
        {
            Tensor half = result.to_dtype(Dtype::Float16, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            return half;
        }
        return result;
    }
#endif
    (void)stream;
    std::copy(host_data, host_data + static_cast<std::ptrdiff_t>(result.size_), result.data());
    if (dtype == Dtype::Float16)
    {
        return result.to_dtype(Dtype::Float16, 0);
    }
    return result;
}

} // namespace dl
