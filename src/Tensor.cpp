#include "DeepLearnLib/Tensor.hpp"
#include <numeric>
#include <stdexcept>
#include <string>

namespace dl
{

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

static auto calculate_size(const std::vector<int>& shape) -> int
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

Tensor::Tensor(std::vector<int> shape, Device device_type)
    : shape_(std::move(shape))
    , device_(device_type)
    , size_(calculate_size(shape_))
{
    compute_strides();
#if DEEPLEARNLIB_ENABLE_CUDA
    if (device_ == Device::GPU)
    {
        int device_count{ 0 };
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        if (device_count == 0)
        {
            throw std::runtime_error("No CUDA-capable devices found");
        }

        void* gpu_pointer{ nullptr };
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
    strides_.resize(shape_.size());
    if (shape_.empty())
    {
        return;
    }
    strides_.back() = 1;
    for (int dim_idx = static_cast<int>(shape_.size()) - 2; dim_idx >= 0; --dim_idx)
    {
        strides_[dim_idx] = strides_[dim_idx + 1] * shape_[dim_idx + 1];
    }
}

auto Tensor::is_contiguous() const -> bool
{
    if (shape_.empty())
    {
        return true;
    }

    int expected_stride{ 1 };
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

    const int k_left{ shape_.back() };
    const int k_right{ other.shape_.front() };
    if (k_left != k_right)
    {
        throw std::runtime_error("matmul inner dimensions must match (" + std::to_string(k_left) + " vs " +
                                 std::to_string(k_right) + ")");
    }
    if (k_left <= 0)
    {
        throw std::runtime_error("matmul inner dimension must be positive");
    }

    const int K{ k_left };
    const int M{ static_cast<int>(size_ / static_cast<size_t>(K)) };
    const int N{ static_cast<int>(other.size_ / static_cast<size_t>(K)) };

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
    const float alpha{ 1.0F };
    const float beta{ 0.0F };
    CHECK_CUBLAS(cublasSgemm(get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, other.data(), N, data(), K,
                             &beta, result.data(), N));

    return result;
#endif
}

} // namespace dl
