#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace
{

constexpr int kThreads = 256;

// Loads an activation element as float. The template keeps one kernel per storage
// type, while the arithmetic still runs in float.
template <typename Act>
__device__ auto load_act(const Act* pointer, int index) -> float
{
    return pointer[index];
}

// Stores a float result into the activation buffer. A separate function so forward
// and backward do not repeat the cast on the store.
template <typename Act>
__device__ auto store_act(Act* pointer, int index, float value) -> void
{
    pointer[index] = value;
}

// Computes LeakyReLU element by element. Positive values pass through unchanged
// and negative values are multiplied by the slope, with no second pass over memory.
template <typename Act>
__global__ void leaky_forward_kernel(const Act* input, Act* output, float slope, int total)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= total)
    {
        return;
    }
    const float value = load_act(input, index);
    // Positive side is the identity. The negative side is scaled by slope in the same store.
    store_act(output, index, value > 0.0F ? value : value * slope);
}

// Multiplies the output gradient by 1 or by the slope, according to the sign of
// the input. The derivative is constant on each half-axis, so the input saved by forward is enough.
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
    // Derivative is 1 where the cached input is positive, and slope elsewhere.
    store_act(grad_input, index, incoming * (value > 0.0F ? 1.0F : slope));
}

// Computes the block grid of the elementwise kernel. Rounding up by the element
// count and kThreads covers the last, partial block.
auto elementwise_grid(int count) -> dim3
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// Rejects a host tensor or a null device pointer. The layer does not copy data
// onto the GPU inside forward.
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

// Stores the slope of the negative half-axis and sets the device to GPU. The slope
// is fixed; it is not a trained parameter.
LeakyReLU::LeakyReLU(float slope_val)
    : slope_(slope_val)
{
    device_ = dl::Device::GPU;
}

// Launches the LeakyReLU kernel and returns a view of the output cache. The input
// stays in the cache because backward needs its sign, and ensure does not allocate when the shape is unchanged.
auto LeakyReLU::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("LeakyReLU_Forward");
    const dl::StreamGuard stream_guard(stream);
    require_gpu(input_tensor, "LeakyReLU::forward input");

    // Backward reads the sign from this view. The view does not copy the input.
    input_cache_ = input_tensor.as_view();
    input_cache_ready_ = true;

    dl::Tensor& output = dl::Tensor::ensure(output_cache_, input_tensor.get_shape(), dl::Device::GPU,
        input_tensor.get_dtype());
    const int total = static_cast<int>(input_tensor.get_size());
    if (total == 0)
    {
        return output.as_view();
    }

    leaky_forward_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(input_tensor.data(), output.data(), slope_,
        total);
    CHECK_CUDA(cudaGetLastError());
    return output.as_view();
}

// Passes the gradient through the LeakyReLU derivative using the cached input.
// The cache flag is cleared so a later backward without a forward cannot reuse a stale sign.
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
        // Drop the cache even when there is nothing to differentiate.
        input_cache_ready_ = false;
        return grad_input.as_view();
    }

    leaky_backward_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(output_error_derivative.data(),
        input_cache_->data(), grad_input.data(), slope_, total);
    CHECK_CUDA(cudaGetLastError());
    // A later backward without a new forward must not reuse this sign.
    input_cache_ready_ = false;
    return grad_input.as_view();
}
