#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/Nvtx.hpp"

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
    thrust::transform(thrust::cuda::par.on(dl::current_stream()), thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(tensor.get_size())), out,
        UniformFill { low, high, seed });
    CHECK_CUDA(cudaGetLastError());
}

auto fill_constant(dl::Tensor& tensor, float value) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    auto out = thrust::device_pointer_cast(tensor.data());
    thrust::fill(thrust::cuda::par.on(dl::current_stream()), out, out + static_cast<std::ptrdiff_t>(tensor.get_size()), value);
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
    if (src.get_dtype() == dst.get_dtype())
    {
        dl::memcpy_d2d_on_current(dst.data(), src.data(), src.nbytes());
        return;
    }
    const dl::Tensor converted = src.to_dtype(dst.get_dtype(), dl::current_stream());
    dl::memcpy_d2d_on_current(dst.data(), converted.data(), dst.nbytes());
}

auto require_rank2(const dl::Tensor& tensor, int expected_cols, const char* name) -> void
{
    require_gpu(tensor, name);
    if (tensor.get_shape().size() != 2)
    {
        throw std::runtime_error(std::string(name) + " must be rank-2 [batch, features], got "
            + tensor.describe());
    }
    if (tensor.get_shape()[1] != expected_cols)
    {
        throw std::runtime_error(std::string(name) + " has an unexpected feature dimension (expected "
            + std::to_string(expected_cols) + ", got " + tensor.describe() + ")");
    }
}

auto fullyconnected_weight_shape(int input_size, int output_size) -> std::vector<int>
{
    if (input_size <= 0 || output_size <= 0)
    {
        throw std::runtime_error("FullyConnected requires positive input and output sizes");
    }
    return { input_size, output_size };
}

} // namespace

FullyConnected::FullyConnected(int input_size, int output_size, float inertia_val)
    : weights_(fullyconnected_weight_shape(input_size, output_size), dl::Device::GPU)
    , biases_({ 1, output_size }, dl::Device::GPU)
    , weights_gradient_({ input_size, output_size }, dl::Device::GPU)
    , biases_gradient_({ 1, output_size }, dl::Device::GPU)
    , input_size_(input_size)
    , output_size_(output_size)
    , inertia_(inertia_val)
{
    device_ = dl::Device::GPU;
    const float bound = std::sqrt(dl::safe_inv(static_cast<float>(input_size_)));
    fill_uniform(weights_, -bound, bound, 0xF00DULL);
    fill_uniform(biases_, -bound, bound, 0xBEEFULL);
    fill_constant(weights_gradient_, 0.0F);
    fill_constant(biases_gradient_, 0.0F);

    if (dl::compute_dtype() == dl::Dtype::Float16)
    {
        weights_ = weights_.to_dtype(dl::Dtype::Float16);
        biases_ = biases_.to_dtype(dl::Dtype::Float16);
        weights_gradient_ = weights_gradient_.to_dtype(dl::Dtype::Float16);
        biases_gradient_ = biases_gradient_.to_dtype(dl::Dtype::Float16);
    }
}

auto FullyConnected::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("FullyConnected_Forward");
    const dl::StreamGuard stream_guard(stream);
    require_rank2(input_tensor, input_size_, "FullyConnected::forward input");

    if (input_tensor.get_dtype() != weights_.get_dtype())
    {
        input_cache_ = input_tensor.to_dtype(weights_.get_dtype(), stream);
    }
    else
    {
        input_cache_ = input_tensor.as_view();
    }
    input_cache_ready_ = true;

    dl::Tensor& output = dl::Tensor::ensure(output_cache_, { input_cache_->get_shape()[0], output_size_ },
        dl::Device::GPU, weights_.get_dtype());
    input_cache_->matmul_into(weights_, output);
    output.add_row_(biases_);
    return output.as_view();
}

auto FullyConnected::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("FullyConnected_Backward");
    const dl::StreamGuard stream_guard(stream);
    if (!input_cache_ready_ || !input_cache_.has_value())
    {
        throw std::runtime_error("FullyConnected::backward requires a preceding forward pass");
    }
    require_rank2(output_error_derivative, output_size_, "FullyConnected::backward grad_output");
    if (output_error_derivative.get_shape()[0] != input_cache_->get_shape()[0])
    {
        throw std::runtime_error("FullyConnected::backward batch size does not match the cached input");
    }

    const dl::Tensor* grad_output = &output_error_derivative;
    dl::Tensor converted_grad;
    if (output_error_derivative.get_dtype() != weights_.get_dtype())
    {
        converted_grad = output_error_derivative.to_dtype(weights_.get_dtype(), stream);
        grad_output = &converted_grad;
    }

    input_cache_->matmul_into(*grad_output, weights_gradient_, true, false, inertia_);
    biases_gradient_.add_sum_rows_(*grad_output, inertia_);

    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, input_cache_->get_shape(), dl::Device::GPU,
        weights_.get_dtype());
    grad_output->matmul_into(weights_, grad_input, false, true, 0.0F);
    input_cache_ready_ = false;
    return grad_input.as_view();
}

void FullyConnected::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("FullyConnected_Step");
    const dl::StreamGuard stream_guard(stream);
    weights_.sgd_update_(weights_gradient_, scaled_learning_rate(), kWeightDecay, parameter_clip_bound());
    biases_.sgd_update_(biases_gradient_, scaled_learning_rate(), kWeightDecay, parameter_clip_bound());
}

void FullyConnected::clip_gradients(float abs_bound, cudaStream_t stream)
{
    const dl::StreamGuard stream_guard(stream);
    if (abs_bound <= 0.0F)
    {
        return;
    }
    weights_gradient_.clamp_(-abs_bound, abs_bound);
    biases_gradient_.clamp_(-abs_bound, abs_bound);
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
