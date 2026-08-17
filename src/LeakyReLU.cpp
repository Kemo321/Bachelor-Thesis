#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <cstddef>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace
{

constexpr int kThreads = 256;

template <typename Act>
__device__ auto load_act(const Act* pointer, int index) -> float
{
    if constexpr (std::is_same_v<Act, __half>)
    {
        return __half2float(pointer[index]);
    }
    else
    {
        return pointer[index];
    }
}

template <typename Act>
__device__ auto store_act(Act* pointer, int index, float value) -> void
{
    if constexpr (std::is_same_v<Act, __half>)
    {
        pointer[index] = __float2half(value);
    }
    else
    {
        pointer[index] = value;
    }
}

template <typename Act>
__global__ void leaky_forward_kernel(const Act* input, Act* output, float slope, int total)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= total)
    {
        return;
    }
    const float value = load_act(input, index);
    store_act(output, index, value > 0.0F ? value : value * slope);
}

template <typename Act>
__global__ void leaky_backward_kernel(const Act* grad_output, const Act* input, Act* grad_input, float slope, int total)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= total)
    {
        return;
    }
    const float incoming = load_act(grad_output, index);
    const float value = load_act(input, index);
    store_act(grad_input, index, incoming * (value > 0.0F ? 1.0F : slope));
}

auto elementwise_grid(int count) -> dim3
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

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

    input_cache_ = input_tensor.as_view();
    input_cache_ready_ = true;

    dl::Tensor& output = dl::Tensor::ensure(output_cache_, input_tensor.get_shape(), dl::Device::GPU,
        input_tensor.get_dtype());
    const int total = static_cast<int>(input_tensor.get_size());
    if (total == 0)
    {
        return output.as_view();
    }

    if (input_tensor.get_dtype() == dl::Dtype::Float16)
    {
        leaky_forward_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(input_tensor.half_data(),
            output.half_data(), slope_, total);
    }
    else
    {
        leaky_forward_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(input_tensor.data(), output.data(),
            slope_, total);
    }
    CHECK_CUDA(cudaGetLastError());
    return output.as_view();
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
    if (output_error_derivative.get_dtype() != input_cache_->get_dtype())
    {
        throw std::runtime_error("LeakyReLU::backward grad_output dtype does not match the cached input");
    }

    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, input_cache_->get_shape(), dl::Device::GPU,
        input_cache_->get_dtype());
    const int total = static_cast<int>(output_error_derivative.get_size());
    if (total == 0)
    {
        input_cache_ready_ = false;
        return grad_input.as_view();
    }

    if (input_cache_->get_dtype() == dl::Dtype::Float16)
    {
        leaky_backward_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(output_error_derivative.half_data(),
            input_cache_->half_data(), grad_input.half_data(), slope_, total);
    }
    else
    {
        leaky_backward_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(output_error_derivative.data(),
            input_cache_->data(), grad_input.data(), slope_, total);
    }
    CHECK_CUDA(cudaGetLastError());
    input_cache_ready_ = false;
    return grad_input.as_view();
}
