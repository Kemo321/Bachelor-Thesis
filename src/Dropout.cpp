#include "DeepLearnLib/Dropout.hpp"
#include "DeepLearnLib/Nvtx.hpp"
#include "DeepLearnLib/SafeMath.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace
{

constexpr int kThreads = 256;

struct BernoulliMask
{
    float keep_probability;
    float scale;
    unsigned long long seed;

    // Draws a Bernoulli threshold from a hash of the index and the seed. A kept
    // element stores the inverted-dropout scale, a dropped element stores zero, with no cuRAND state on the device.
    __host__ __device__ auto operator()(int index) const -> float
    {
        unsigned long long hash = seed + (static_cast<unsigned long long>(index) + 1ULL) * 0x9E3779B97F4A7C15ULL;
        hash ^= hash >> 30U;
        hash *= 0xBF58476D1CE4E5B9ULL;
        hash ^= hash >> 27U;
        hash *= 0x94D049BB133111EBULL;
        hash ^= hash >> 31U;
        const float unit = static_cast<float>(hash & 0xFFFFFFULL) / static_cast<float>(0x1000000ULL);
        // Bernoulli keep test. A kept element stores the inverted-dropout scale; a dropped element stores 0.
        return unit < keep_probability ? scale : 0.0F;
    }
};

// Loads an activation element as float. The same load serves the input and gradient
// dtypes, and the multiply by the mask is in float anyway.
template <typename Act>
__device__ auto load_act(const Act* pointer, int index) -> float
{
    return pointer[index];
}

// Stores a float result into the activation buffer. Forward and backward share one
// store so the cast is not duplicated.
template <typename Act>
__device__ auto store_act(Act* pointer, int index, float value) -> void
{
    pointer[index] = value;
}

// Fills the dropout mask on the GPU. The mask is a separate buffer because the same
// pattern has to reach backward.
__global__ void dropout_mask_kernel(float* mask, BernoulliMask generator, int total)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < total)
    {
        mask[index] = generator(index);
    }
}

// Multiplies a tensor by the mask elementwise. The same kernel serves forward and
// backward, because both are that multiply.
template <typename Act>
__global__ void dropout_apply_kernel(const Act* input, const float* mask, Act* output, int total)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < total)
    {
        store_act(output, index, load_act(input, index) * mask[index]);
    }
}

// Computes the block grid of the elementwise kernel. Rounding up by the element
// count and kThreads covers the last, partial block.
auto elementwise_grid(int count) -> dim3
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// Rejects a host tensor or a null device pointer. The mask is created on the GPU
// and there is no host-to-device copy here.
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

// Checks that the drop probability lies in [0, 1), and sets the mask seed.
// A value of 1 would zero the whole tensor, and the scale 1/(1-p) would be zero.
Dropout::Dropout(float probability)
    : probability_(probability)
    , seed_(0xD10U)
{
    if (probability_ < 0.0F || probability_ >= 1.0F)
    {
        throw std::runtime_error("Dropout probability must be in [0, 1)");
    }
    device_ = dl::Device::GPU;
}

// In train, draws a Bernoulli mask and applies it to the input; in eval, returns a
// view of the input. The inverted-dropout scale lives in the mask, so inference does not multiply separately.
auto Dropout::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Dropout_Forward");
    const dl::StreamGuard stream_guard(stream);
    require_gpu(input_tensor, "Dropout::forward input");

    // Eval is the identity: the inverted-dropout scale is already in the mask from training.
    if (!is_training_)
    {
        mask_ready_ = false;
        return input_tensor.as_view();
    }

    const float keep_probability = 1.0F - probability_;
    // Kept elements carry 1/(1-p); dropped elements stay 0. A new seed on every call.
    const float scale = dl::safe_inv(keep_probability);
    ++seed_;

    dl::Tensor& mask = dl::Tensor::ensure(mask_, input_tensor.get_shape(), dl::Device::GPU, dl::Dtype::Float32);
    dl::Tensor& output = dl::Tensor::ensure(output_cache_, input_tensor.get_shape(), dl::Device::GPU,
        input_tensor.get_dtype());
    const int total = static_cast<int>(input_tensor.get_size());
    if (total == 0)
    {
        mask_ready_ = true;
        return output.as_view();
    }

    dropout_mask_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(mask.data(),
        BernoulliMask { keep_probability, scale, seed_ }, total);
    CHECK_CUDA(cudaGetLastError());
    dropout_apply_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(input_tensor.data(), mask.data(),
        output.data(), total);
    CHECK_CUDA(cudaGetLastError());
    mask_ready_ = true;
    return output.as_view();
}

// In train, multiplies the gradient by the same mask as forward. In eval, or without
// a mask, the gradient is returned unchanged, because forward zeroed nothing.
auto Dropout::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Dropout_Backward");
    const dl::StreamGuard stream_guard(stream);
    require_gpu(output_error_derivative, "Dropout::backward grad_output");
    // Without a mask the forward was the identity, so the gradient is not multiplied.
    if (!is_training_ || !mask_ready_ || !mask_.has_value())
    {
        return output_error_derivative.as_view();
    }
    if (output_error_derivative.get_size() != mask_->get_size())
    {
        throw std::runtime_error("Dropout::backward grad_output size does not match the cached mask");
    }

    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, output_error_derivative.get_shape(), dl::Device::GPU,
        output_error_derivative.get_dtype());
    const int total = static_cast<int>(output_error_derivative.get_size());
    if (total == 0)
    {
        mask_ready_ = false;
        return grad_input.as_view();
    }

    // Same Bernoulli mask as forward, inverted-dropout scale included.
    dropout_apply_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(output_error_derivative.data(),
        mask_->data(), grad_input.data(), total);
    CHECK_CUDA(cudaGetLastError());
    mask_ready_ = false;
    return grad_input.as_view();
}
