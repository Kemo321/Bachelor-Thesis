#include "DeepLearnLib/Tensor.hpp"
#include <cstddef>
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
#endif

} // namespace

#if DEEPLEARNLIB_ENABLE_CUDA
CublasContext::CublasContext()
{
    CHECK_CUBLAS(cublasCreate(&handle_));
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
#endif

Tensor::Tensor(std::vector<int> shape, Device device_type)
    : shape_(std::move(shape))
    , device_(device_type)
    , size_(calculate_size(shape_))
{
    compute_strides();
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
        CHECK_CUDA(cudaMalloc(&gpu_pointer, size_ * sizeof(float)));
        data_ = std::shared_ptr<float>(static_cast<float*>(gpu_pointer), CudaDeleter());
    }
    else
    {
        data_ = std::shared_ptr<float>(new float[size_](), CpuDeleter());
    }
#else
    if (device_ == Device::GPU)
    {
        throw std::runtime_error("CUDA support is not enabled");
    }
    data_ = std::shared_ptr<float>(new float[size_](), CpuDeleter());
#endif
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Tensor::Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data_ptr, Device device_type)
    : shape_(std::move(shape))
    , strides_(std::move(strides))
    , device_(device_type)
    , size_(calculate_size(shape_))
    , data_(std::move(data_ptr))
{
}

auto Tensor::get_shape() const -> const std::vector<int>&
{
    return shape_;
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
}

auto Tensor::matmul(const Tensor& other) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("matmul requires CUDA/cuBLAS support");
#else
    if (device_ != Device::GPU || other.device_ != Device::GPU)
    {
        throw std::runtime_error("matmul requires both tensors to reside on the GPU");
    }
    if (data_.get() == nullptr || other.data_.get() == nullptr)
    {
        throw std::runtime_error("matmul requires valid device pointers");
    }
    if (shape_.empty() || other.shape_.empty())
    {
        throw std::runtime_error("matmul requires non-scalar tensors");
    }
    if (!is_contiguous() || !other.is_contiguous())
    {
        throw std::runtime_error("matmul requires contiguous row-major tensors");
    }

    const int k_left { shape_.back() };
    const int k_right { other.shape_.front() };
    if (k_left != k_right)
    {
        throw std::runtime_error("matmul inner dimensions must match (" + std::to_string(k_left) + " vs " + std::to_string(k_right) + ")");
    }
    if (k_left <= 0)
    {
        throw std::runtime_error("matmul inner dimension must be positive");
    }

    const int K { k_left };
    const int M { static_cast<int>(size_ / static_cast<size_t>(K)) };
    const int N { static_cast<int>(other.size_ / static_cast<size_t>(K)) };

    std::vector<int> result_shape;
    result_shape.reserve(shape_.size() + other.shape_.size() - 2);
    result_shape.insert(result_shape.end(), shape_.begin(), shape_.end() - 1);
    result_shape.insert(result_shape.end(), other.shape_.begin() + 1, other.shape_.end());
    if (result_shape.empty())
    {
        result_shape.push_back(1);
    }

    Tensor result(result_shape, Device::GPU);

    if (M == 0 || N == 0)
    {
        return result;
    }

    // Row-major C = A * B is computed as column-major C^T = B^T * A^T.
    // Interpreting row-major storage as column-major therefore means swapping A/B
    // and using CUBLAS_OP_N for both operands.
    const float alpha { 1.0F };
    const float beta { 0.0F };
    CHECK_CUBLAS(cublasSgemm(get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, other.data(), N, data(), K,
        &beta, result.data(), N));

    return result;
#endif
}

auto Tensor::operator+(const Tensor& other) const -> Tensor
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("operator+ requires CUDA/Thrust support");
#else
    ensure_binary_op(other, "operator+");
    Tensor result(shape_, Device::GPU);
    if (size_ == 0)
    {
        return result;
    }

    auto lhs = thrust::device_pointer_cast(data());
    auto rhs = thrust::device_pointer_cast(other.data());
    auto out = thrust::device_pointer_cast(result.data());
    thrust::transform(thrust::device, lhs, lhs + static_cast<std::ptrdiff_t>(size_), rhs, out, thrust::plus<float>());
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
    Tensor result(shape_, Device::GPU);
    if (size_ == 0)
    {
        return result;
    }

    auto lhs = thrust::device_pointer_cast(data());
    auto rhs = thrust::device_pointer_cast(other.data());
    auto out = thrust::device_pointer_cast(result.data());
    thrust::transform(thrust::device, lhs, lhs + static_cast<std::ptrdiff_t>(size_), rhs, out, thrust::minus<float>());
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
    Tensor result(shape_, Device::GPU);
    if (size_ == 0)
    {
        return result;
    }

    auto lhs = thrust::device_pointer_cast(data());
    auto rhs = thrust::device_pointer_cast(other.data());
    auto out = thrust::device_pointer_cast(result.data());
    thrust::transform(thrust::device, lhs, lhs + static_cast<std::ptrdiff_t>(size_), rhs, out,
        thrust::multiplies<float>());
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

    Tensor result(shape_, Device::GPU);
    if (size_ == 0)
    {
        return result;
    }

    auto in = thrust::device_pointer_cast(data());
    auto out = thrust::device_pointer_cast(result.data());
    thrust::transform(thrust::device, in, in + static_cast<std::ptrdiff_t>(size_), out,
        thrust::placeholders::_1 * scalar);
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

    Tensor result(shape_, Device::GPU);
    if (size_ == 0)
    {
        return result;
    }

    auto in = thrust::device_pointer_cast(data());
    auto out = thrust::device_pointer_cast(result.data());
    thrust::transform(thrust::device, in, in + static_cast<std::ptrdiff_t>(size_), out,
        thrust::placeholders::_1 + scalar);
    CHECK_CUDA(cudaGetLastError());
    return result;
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

    Tensor result(shape_, Device::GPU);
    if (size_ == 0)
    {
        return result;
    }

    auto in = thrust::device_pointer_cast(data());
    auto out = thrust::device_pointer_cast(result.data());
    thrust::transform(thrust::device, in, in + static_cast<std::ptrdiff_t>(size_), out, ClampValue { lo, hi });
    CHECK_CUDA(cudaGetLastError());
    return result;
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

    Tensor result({ 1 }, Device::GPU);
    float total { 0.0F };
    if (size_ > 0)
    {
        auto begin = thrust::device_pointer_cast(data());
        total = thrust::reduce(thrust::device, begin, begin + static_cast<std::ptrdiff_t>(size_), 0.0F,
            thrust::plus<float>());
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaMemcpy(result.data(), &total, sizeof(float), cudaMemcpyHostToDevice));
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
    return Tensor(std::move(shape), std::move(strides), data_, device_);
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
    Tensor result({ cols, rows }, Device::GPU);
    if (size_ == 0 || rows == 0 || cols == 0)
    {
        return result;
    }

    thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(size_)),
        Transpose2D { data(), result.data(), rows, cols });
    CHECK_CUDA(cudaGetLastError());
    return result;
#endif
}

auto Tensor::zeros_like(const Tensor& other) -> Tensor
{
    Tensor result(other.shape_, other.device_);
#if DEEPLEARNLIB_ENABLE_CUDA
    if (result.device_ == Device::GPU && result.size_ > 0)
    {
        auto out = thrust::device_pointer_cast(result.data());
        thrust::fill(thrust::device, out, out + static_cast<std::ptrdiff_t>(result.size_), 0.0F);
        CHECK_CUDA(cudaGetLastError());
    }
#endif
    return result;
}

auto Tensor::to_host() const -> std::vector<float>
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
        CHECK_CUDA(cudaMemcpy(host.data(), data_.get(), size_ * sizeof(float), cudaMemcpyDeviceToHost));
        return host;
    }
#endif
    std::copy(data_.get(), data_.get() + static_cast<std::ptrdiff_t>(size_), host.begin());
    return host;
}

auto Tensor::from_host(const std::vector<int>& shape, const std::vector<float>& host_data, Device device) -> Tensor
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
    return from_host(shape, host_data.data(), device);
}

auto Tensor::from_host(const std::vector<int>& shape, const float* host_data, Device device) -> Tensor
{
    Tensor result(shape, device);
    if (result.size_ == 0)
    {
        return result;
    }
    if (host_data == nullptr)
    {
        throw std::runtime_error("from_host requires a non-null host pointer");
    }
#if DEEPLEARNLIB_ENABLE_CUDA
    if (device == Device::GPU)
    {
        CHECK_CUDA(cudaMemcpy(result.data(), host_data, result.size_ * sizeof(float), cudaMemcpyHostToDevice));
        return result;
    }
#endif
    std::copy(host_data, host_data + static_cast<std::ptrdiff_t>(result.size_), result.data());
    return result;
}

} // namespace dl
