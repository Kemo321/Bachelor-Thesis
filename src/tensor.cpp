#include "DeepLearnLib/tensor.hpp"
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
        auto gpu_data = static_cast<gsl::owner<float*>>(nullptr);
        cudaMalloc(&gpu_data, size_ * sizeof(float));
        data_ = std::shared_ptr<float>(gpu_data, CudaDeleter());
    }
    else
    {
        auto cpu_data = static_cast<gsl::owner<float*>>(new float[size_]());
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

// auto Tensor::get_shape() const -> const std::vector<int>&
// {
//     return shape_;
// }

// auto Tensor::get_strides() const -> const std::vector<int>&
// {
//     return strides_;
// }

// auto Tensor::get_size() const -> size_t
// {
//     return size_;
// }

// auto Tensor::get_device() const -> Device
// {
//     return device_;
// }

// auto Tensor::get_data() const -> const float*
// {
//     return data_.get();
// }

// auto data() -> float*;
// auto data() const -> const float*;

// auto to_device(Device target_device) -> void;

// auto operator+(const Tensor& other) const -> Tensor;
// auto operator-(const Tensor& other) const -> Tensor;
// auto operator*(const Tensor& other) const -> Tensor;
// auto operator/(const Tensor& other) const -> Tensor;
// auto dot(const Tensor& other) const -> Tensor;

// auto operator*(float scalar) const -> Tensor;
// auto operator/(float scalar) const -> Tensor;
// auto operator+(float scalar) const -> Tensor;
// auto operator-(float scalar) const -> Tensor;

// auto sum() const -> float;
// auto mean() const -> float;
// auto max() const -> float;
// auto min() const -> float;

// auto reshape(const std::vector<int>& new_shape) -> void;
// auto transpose() const -> Tensor;
// auto flatten() const -> Tensor;

// auto relu() const -> Tensor;
// auto sigmoid() const -> Tensor;
// auto softmax(int axis = -1) const -> Tensor;
// auto conv2d(const Tensor& kernel, int stride = 1, int padding = 0) const -> Tensor;
// auto max_pool2d(int pool_size, int stride) const -> Tensor;
// auto leaky_relu(float alpha = 0.01f) const -> Tensor;

// static auto concat(const std::vector<Tensor>& tensors, int axis = 0) -> Tensor;

// auto at(const std::vector<int>& indices) const -> float;
// auto to_host() const -> std::vector<float>;

// static auto zeros(const std::vector<int>& shape, Device device = Device::CPU) -> Tensor;
// static auto ones(const std::vector<int>& shape, Device device = Device::CPU) -> Tensor;
// static auto random(const std::vector<int>& shape, Device device = Device::CPU) -> Tensor;

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
