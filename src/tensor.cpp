#include "DeepLearnLib/tensor.hpp"
#include <gsl/gsl>
#include <numeric>

namespace dl
{

static auto calculate_size(const std::vector<int>& shape) -> int
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

Tensor::Tensor(std::vector<int> shape, Device device)
    : shape_(std::move(shape))
    , strides_()
    , device_(device)
    , size_(0)
{
    compute_strides();
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    if (device_ == Device::GPU)
    {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0)
        {
            throw std::runtime_error("No CUDA-capable devices found");
        }
        auto* gpu_data = gsl::owner<float*>(nullptr);
        cudaMalloc(&gpu_data, size_ * sizeof(float));
        data_ = std::shared_ptr<float>(gpu_data, CudaDeleter());
    }
    else
    {
        auto* cpu_data = gsl::owner<float*>(new float[size_]());
        data_ = std::shared_ptr<float>(cpu_data, CpuDeleter());
    }
}

Tensor::Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data, Device device)
    : shape_(std::move(shape))
    , strides_(std::move(strides))
    , data_(std::move(data))
    , device_(device)
    , size_(calculate_size(shape_))
{
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
