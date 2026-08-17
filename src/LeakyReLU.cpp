#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/Nvtx.hpp"

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
    dl::memcpy_d2d_on_current(dst.data(), src.data(), src.nbytes());
}

} // namespace

LeakyReLU::LeakyReLU(float slope_val)
    : slope_(slope_val)
{
    device_ = dl::Device::GPU;
}

auto LeakyReLU::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("LeakyReLU_Forward");
    const dl::StreamGuard stream_guard(stream);
    require_gpu(input_tensor, "LeakyReLU::forward input");

    const dl::Dtype dtype = input_tensor.get_dtype();
    const dl::Tensor* input = &input_tensor;
    dl::Tensor converted;
    if (dtype == dl::Dtype::Float16)
    {
        converted = input_tensor.to_dtype(dl::Dtype::Float32, stream);
        input = &converted;
    }

    dl::Tensor& cached = dl::Tensor::ensure(input_cache_, input->get_shape(), dl::Device::GPU, input->get_dtype());
    copy_same_size(cached, *input, "LeakyReLU::forward input cache");
    input_cache_ready_ = true;

    dl::Tensor output(input->get_shape(), dl::Device::GPU, input->get_dtype());
    if (input->get_size() == 0)
    {
        if (dtype == dl::Dtype::Float16)
        {
            return output.to_dtype(dl::Dtype::Float16, stream);
        }
        return output;
    }

    auto in = thrust::device_pointer_cast(input->data());
    auto out = thrust::device_pointer_cast(output.data());
    thrust::transform(thrust::cuda::par.on(dl::current_stream()), in, in + static_cast<std::ptrdiff_t>(input->get_size()),
        out, LeakyReluForward { slope_ });
    CHECK_CUDA(cudaGetLastError());
    if (dtype == dl::Dtype::Float16)
    {
        return output.to_dtype(dl::Dtype::Float16, stream);
    }
    return output;
}

auto LeakyReLU::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("LeakyReLU_Backward");
    const dl::StreamGuard stream_guard(stream);
    if (!input_cache_ready_ || !input_cache_.has_value())
    {
        throw std::runtime_error("LeakyReLU::backward requires a preceding forward pass");
    }
    require_gpu(output_error_derivative, "LeakyReLU::backward grad_output");
    if (output_error_derivative.get_size() != input_cache_->get_size())
    {
        throw std::runtime_error("LeakyReLU::backward grad_output size does not match the cached input");
    }

    const dl::Dtype dtype = output_error_derivative.get_dtype();
    const dl::Tensor* grad_output = &output_error_derivative;
    dl::Tensor converted_grad;
    if (dtype == dl::Dtype::Float16)
    {
        converted_grad = output_error_derivative.to_dtype(dl::Dtype::Float32, stream);
        grad_output = &converted_grad;
    }

    dl::Tensor grad_input(input_cache_->get_shape(), dl::Device::GPU, input_cache_->get_dtype());
    if (grad_output->get_size() == 0)
    {
        input_cache_ready_ = false;
        if (dtype == dl::Dtype::Float16)
        {
            return grad_input.to_dtype(dl::Dtype::Float16, stream);
        }
        return grad_input;
    }

    auto dy = thrust::device_pointer_cast(grad_output->data());
    auto x = thrust::device_pointer_cast(input_cache_->data());
    auto dx = thrust::device_pointer_cast(grad_input.data());
    thrust::transform(thrust::cuda::par.on(dl::current_stream()), dy, dy + static_cast<std::ptrdiff_t>(grad_output->get_size()),
        x, dx, LeakyReluBackward { slope_ });
    CHECK_CUDA(cudaGetLastError());
    input_cache_ready_ = false;
    if (dtype == dl::Dtype::Float16)
    {
        return grad_input.to_dtype(dl::Dtype::Float16, stream);
    }
    return grad_input;
}
