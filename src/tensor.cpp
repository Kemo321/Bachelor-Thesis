#include "DeepLearnLib/tensor.hpp"
#include <numeric>
#include <stdexcept>

namespace dl
{

static auto calculate_size(const std::vector<int>& shape) -> int
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

Tensor::Tensor(std::vector<int> shape, Device device)
    : shape_(std::move(shape))
    , device_(device)
    , size_(calculate_size(shape_))
{
    compute_strides();

    if (device_ == Device::GPU)
    {
        int device_count { 0 };
        cudaError_t err { cudaSuccess };

        err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0)
        {
            throw std::runtime_error("No CUDA-capable devices found");
        }

        float* gpu_ptr { nullptr };

        // 1. Usunięto redundantny static_cast (size_ to już size_t)
        // 2. Dodano NOLINT, aby linter nie czepiał się reinterpret_cast wymaganego przez CUDA
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        err = cudaMalloc(reinterpret_cast<void**>(&gpu_ptr), size_ * sizeof(float));

        if (err != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed");
        }

        data_ = std::shared_ptr<float>(gpu_ptr, CudaDeleter());
    }
    else
    {
        data_ = std::shared_ptr<float>(new float[size_](), CpuDeleter());
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Tensor::Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data, Device device)
    : shape_(std::move(shape))
    , strides_(std::move(strides))
    , data_(std::move(data))
    , device_(device)
    , size_(calculate_size(shape_))
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
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i)
    {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}
} // namespace dl
