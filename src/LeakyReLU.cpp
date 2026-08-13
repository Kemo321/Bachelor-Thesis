#include "DeepLearnLib/LeakyReLU.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace
{

struct LeakyReluForward
{
    float slope;

    __host__ __device__ auto operator()(float value) const -> float
    {
        return value > 0.0F ? value : value * slope;
    }
};

struct LeakyReluBackward
{
    float slope;

    __host__ __device__ auto operator()(float grad_output, float input_value) const -> float
    {
        return grad_output * (input_value > 0.0F ? 1.0F : slope);
    }
};

auto require_gpu(const dl::Tensor& tensor, const char* name) -> void
{
    if (tensor.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " must reside on the GPU");
    }
    if (tensor.get_size() > 0 && tensor.data() == nullptr)
    {
        throw std::runtime_error(std::string(name) + " has a null device pointer");
    }
}

auto copy_same_size(dl::Tensor& dst, const dl::Tensor& src, const char* name) -> void
{
    if (src.get_size() != dst.get_size())
    {
        throw std::runtime_error(std::string(name) + " tensor size mismatch");
    }
    if (src.get_size() == 0)
    {
        return;
    }
    CHECK_CUDA(cudaMemcpy(dst.data(), src.data(), src.get_size() * sizeof(float), cudaMemcpyDeviceToDevice));
}

} // namespace

LeakyReLU::LeakyReLU(float slope_val)
    : slope_(slope_val)
{
    device_ = dl::Device::GPU;
}

auto LeakyReLU::forward(const dl::Tensor& input_tensor) -> dl::Tensor
{
    require_gpu(input_tensor, "LeakyReLU::forward input");

    input_cache_ = dl::Tensor(input_tensor.get_shape(), dl::Device::GPU);
    copy_same_size(*input_cache_, input_tensor, "LeakyReLU::forward input cache");

    dl::Tensor output(input_tensor.get_shape(), dl::Device::GPU);
    if (input_tensor.get_size() == 0)
    {
        return output;
    }

    auto in = thrust::device_pointer_cast(input_tensor.data());
    auto out = thrust::device_pointer_cast(output.data());
    thrust::transform(thrust::device, in, in + static_cast<std::ptrdiff_t>(input_tensor.get_size()), out,
                      LeakyReluForward{ slope_ });
    CHECK_CUDA(cudaGetLastError());
    return output;
}

auto LeakyReLU::backward(const dl::Tensor& output_error_derivative) -> dl::Tensor
{
    if (!input_cache_.has_value())
    {
        throw std::runtime_error("LeakyReLU::backward requires a preceding forward pass");
    }
    require_gpu(output_error_derivative, "LeakyReLU::backward grad_output");
    if (output_error_derivative.get_size() != input_cache_->get_size())
    {
        throw std::runtime_error("LeakyReLU::backward grad_output size does not match the cached input");
    }

    dl::Tensor grad_input(input_cache_->get_shape(), dl::Device::GPU);
    if (output_error_derivative.get_size() == 0)
    {
        input_cache_.reset();
        return grad_input;
    }

    auto dy = thrust::device_pointer_cast(output_error_derivative.data());
    auto x = thrust::device_pointer_cast(input_cache_->data());
    auto dx = thrust::device_pointer_cast(grad_input.data());
    thrust::transform(thrust::device, dy, dy + static_cast<std::ptrdiff_t>(output_error_derivative.get_size()), x, dx,
                      LeakyReluBackward{ slope_ });
    CHECK_CUDA(cudaGetLastError());
    input_cache_.reset();
    return grad_input;
}
