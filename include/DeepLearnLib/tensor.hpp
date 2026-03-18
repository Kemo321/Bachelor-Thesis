#pragma once

#include <algorithm>
#include <cstddef>
#if DEEPLEARNLIB_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#include <memory>
#include <stdexcept>
#include <vector>

namespace dl
{

enum class Device
{
    CPU,
    GPU
};

#if DEEPLEARNLIB_ENABLE_CUDA
struct CudaDeleter
{
    void operator()(float* ptr) const
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
};

#else
struct CpuDeleter
{
    void operator()(float* ptr) const
    {
        if (ptr)
        {
            delete[] ptr;
        }
    }
};
#endif

class Tensor
{
public:
    explicit Tensor(std::vector<int> shape, Device device = Device::CPU);

    Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data, Device device);

    ~Tensor() = default;

    Tensor(const Tensor&) = delete;
    auto operator=(const Tensor&) -> Tensor& = delete;

    Tensor(Tensor&&) noexcept = default;
    auto operator=(Tensor&&) noexcept -> Tensor& = default;

    // clang-format off
    // cppcheck-suppress unusedFunction
    auto get_shape() const -> const std::vector<int>&;
    auto get_strides() const -> const std::vector<int>&;
    auto get_size() const -> size_t;
    auto get_device() const -> Device;
    auto get_data() const -> const float*;
    // clang-format on

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

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    size_t size_ = 0;
    Device device_;

    std::shared_ptr<float> data_;

    auto compute_strides() -> void;
};

} // namespace dl
