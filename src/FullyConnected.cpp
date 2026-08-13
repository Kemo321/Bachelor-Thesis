#include "DeepLearnLib/FullyConnected.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace
{

constexpr float kWeightDecay = 0.0005F;

struct UniformFill
{
    float low;
    float high;
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
        return low + ((high - low) * unit);
    }
};

auto fill_uniform(dl::Tensor& tensor, float low, float high, unsigned long long seed) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    auto out = thrust::device_pointer_cast(tensor.data());
    thrust::transform(thrust::device, thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(static_cast<int>(tensor.get_size())), out,
                      UniformFill{ low, high, seed });
    CHECK_CUDA(cudaGetLastError());
}

auto fill_constant(dl::Tensor& tensor, float value) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    auto out = thrust::device_pointer_cast(tensor.data());
    thrust::fill(thrust::device, out, out + static_cast<std::ptrdiff_t>(tensor.get_size()), value);
    CHECK_CUDA(cudaGetLastError());
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

auto copy_same_size(dl::Tensor& dst, const dl::Tensor& src, const char* name) -> void
{
    if (src.get_device() != dl::Device::GPU || dst.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " requires GPU tensors");
    }
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

auto require_rank2(const dl::Tensor& tensor, int expected_cols, const char* name) -> void
{
    require_gpu(tensor, name);
    if (tensor.get_shape().size() != 2)
    {
        throw std::runtime_error(std::string(name) + " must be rank-2 [batch, features]");
    }
    if (tensor.get_shape()[1] != expected_cols)
    {
        throw std::runtime_error(std::string(name) + " has an unexpected feature dimension");
    }
}

} // namespace

FullyConnected::FullyConnected(int input_size, int output_size, float inertia_val)
    : weights_({ input_size, output_size }, dl::Device::GPU)
    , biases_({ 1, output_size }, dl::Device::GPU)
    , weights_gradient_({ input_size, output_size }, dl::Device::GPU)
    , biases_gradient_({ 1, output_size }, dl::Device::GPU)
    , input_size_(input_size)
    , output_size_(output_size)
    , inertia_(inertia_val)
{
    if (input_size <= 0 || output_size <= 0)
    {
        throw std::runtime_error("FullyConnected requires positive input and output sizes");
    }

    device_ = dl::Device::GPU;
    const float bound = std::sqrt(1.0F / static_cast<float>(input_size_));
    fill_uniform(weights_, -bound, bound, 0xF00DULL);
    fill_uniform(biases_, -bound, bound, 0xBEEFULL);
    fill_constant(weights_gradient_, 0.0F);
    fill_constant(biases_gradient_, 0.0F);
}

auto FullyConnected::forward(const dl::Tensor& input_tensor) -> dl::Tensor
{
    require_rank2(input_tensor, input_size_, "FullyConnected::forward input");

    input_cache_ = dl::Tensor(input_tensor.get_shape(), dl::Device::GPU);
    copy_same_size(*input_cache_, input_tensor, "FullyConnected::forward input cache");

    const int batch = input_tensor.get_shape()[0];
    dl::Tensor batch_ones({ batch, 1 }, dl::Device::GPU);
    fill_constant(batch_ones, 1.0F);

    return input_tensor.matmul(weights_) + batch_ones.matmul(biases_);
}

auto FullyConnected::backward(const dl::Tensor& output_error_derivative) -> dl::Tensor
{
    if (!input_cache_.has_value())
    {
        throw std::runtime_error("FullyConnected::backward requires a preceding forward pass");
    }
    require_rank2(output_error_derivative, output_size_, "FullyConnected::backward grad_output");
    if (output_error_derivative.get_shape()[0] != input_cache_->get_shape()[0])
    {
        throw std::runtime_error("FullyConnected::backward batch size does not match the cached input");
    }

    const int batch = output_error_derivative.get_shape()[0];
    dl::Tensor ones_row({ 1, batch }, dl::Device::GPU);
    fill_constant(ones_row, 1.0F);

    dl::Tensor cur_weights_grad = input_cache_->transpose().matmul(output_error_derivative);
    dl::Tensor cur_biases_grad = ones_row.matmul(output_error_derivative);

    cur_weights_grad = cur_weights_grad + (weights_ * kWeightDecay);
    cur_biases_grad = cur_biases_grad + (biases_ * kWeightDecay);

    weights_gradient_ = cur_weights_grad + (weights_gradient_ * inertia_);
    biases_gradient_ = cur_biases_grad + (biases_gradient_ * inertia_);

    dl::Tensor grad_input = output_error_derivative.matmul(weights_.transpose());
    input_cache_.reset();
    return grad_input;
}

void FullyConnected::step()
{
    weights_ = weights_ - (weights_gradient_ * learning_rate);
    biases_ = biases_ - (biases_gradient_ * learning_rate);
}

auto FullyConnected::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights", weights_.view(weights_.get_shape()));
    params.emplace("bias", biases_.view(biases_.get_shape()));
    return params;
}

void FullyConnected::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    copy_same_size(weights_, params.at("weights"), "FullyConnected::set_parameters weights");
    copy_same_size(biases_, params.at("bias"), "FullyConnected::set_parameters bias");
}

auto FullyConnected::to(dl::Device device) -> void
{
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("FullyConnected parameters must remain on the GPU");
    }
    device_ = device;
}
