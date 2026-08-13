#include "DeepLearnLib/Dropout.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace
{

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

auto Dropout::forward(const dl::Tensor& input_tensor) -> dl::Tensor
{
    require_gpu(input_tensor, "Dropout::forward input");

    if (!is_training_)
    {
        mask_.reset();
        return input_tensor.view(input_tensor.get_shape());
    }

    const float keep_probability = 1.0F - probability_;
    const float scale = 1.0F / keep_probability;
    ++seed_;

    mask_ = dl::Tensor(input_tensor.get_shape(), dl::Device::GPU);
    if (input_tensor.get_size() == 0)
    {
        return dl::Tensor(input_tensor.get_shape(), dl::Device::GPU);
    }

    auto mask_ptr = thrust::device_pointer_cast(mask_->data());
    thrust::transform(thrust::device, thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(input_tensor.get_size())), mask_ptr,
        BernoulliMask { keep_probability, scale, seed_ });
    CHECK_CUDA(cudaGetLastError());
    return input_tensor * (*mask_);
}

auto Dropout::backward(const dl::Tensor& output_error_derivative) -> dl::Tensor
{
    require_gpu(output_error_derivative, "Dropout::backward grad_output");
    if (!is_training_ || !mask_.has_value())
    {
        return output_error_derivative.view(output_error_derivative.get_shape());
    }
    if (output_error_derivative.get_size() != mask_->get_size())
    {
        throw std::runtime_error("Dropout::backward grad_output size does not match the cached mask");
    }

    dl::Tensor grad_input = output_error_derivative * (*mask_);
    mask_.reset();
    return grad_input;
}
