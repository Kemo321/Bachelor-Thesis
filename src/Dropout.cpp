#include "DeepLearnLib/Dropout.hpp"
#include "DeepLearnLib/Nvtx.hpp"
#include "DeepLearnLib/SafeMath.hpp"

#include <cstddef>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace
{

constexpr int kThreads = 256;

struct BernoulliMask
{
    float keep_probability;
    float scale;
    unsigned long long seed;

    __host__ __device__ auto operator()(int index) const -> float
    {
        unsigned long long hash = seed + (static_cast<unsigned long long>(index) + 1ULL) * 0x9E3779B97F4A7C15ULL;
        hash ^= hash >> 30U;
        hash *= 0xBF58476D1CE4E5B9ULL;
        hash ^= hash >> 27U;
        hash *= 0x94D049BB133111EBULL;
        hash ^= hash >> 31U;
        const float unit = static_cast<float>(hash & 0xFFFFFFULL) / static_cast<float>(0x1000000ULL);
        return unit < keep_probability ? scale : 0.0F;
    }
};

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

__global__ void dropout_mask_kernel(float* mask, BernoulliMask generator, int total)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < total)
    {
        mask[index] = generator(index);
    }
}

template <typename Act>
__global__ void dropout_apply_kernel(const Act* input, const float* mask, Act* output, int total)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < total)
    {
        store_act(output, index, load_act(input, index) * mask[index]);
    }
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

auto Dropout::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Dropout_Forward");
    const dl::StreamGuard stream_guard(stream);
    require_gpu(input_tensor, "Dropout::forward input");

    if (!is_training_)
    {
        mask_ready_ = false;
        return input_tensor.as_view();
    }

    const float keep_probability = 1.0F - probability_;
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
    if (input_tensor.get_dtype() == dl::Dtype::Float16)
    {
        dropout_apply_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(input_tensor.half_data(), mask.data(),
            output.half_data(), total);
    }
    else
    {
        dropout_apply_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(input_tensor.data(), mask.data(),
            output.data(), total);
    }
    CHECK_CUDA(cudaGetLastError());
    mask_ready_ = true;
    return output.as_view();
}

auto Dropout::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Dropout_Backward");
    const dl::StreamGuard stream_guard(stream);
    require_gpu(output_error_derivative, "Dropout::backward grad_output");
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

    if (output_error_derivative.get_dtype() == dl::Dtype::Float16)
    {
        dropout_apply_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(output_error_derivative.half_data(),
            mask_->data(), grad_input.half_data(), total);
    }
    else
    {
        dropout_apply_kernel<<<elementwise_grid(total), kThreads, 0, stream>>>(output_error_derivative.data(),
            mask_->data(), grad_input.data(), total);
    }
    CHECK_CUDA(cudaGetLastError());
    mask_ready_ = false;
    return grad_input.as_view();
}
