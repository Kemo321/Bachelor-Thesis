#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{

constexpr int kFillThreads = 256;

struct UniformFill
{
    float low;
    float high;
    unsigned long long seed;

    // Maps an index onto a value in [low, high) by hashing. Initialisation does not
    // call a host generator and does not synchronise the device.
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

// Applies UniformFill across the tensor elements. The range and seed travel in the
// functor, so the kernel does not read extra buffers.
__global__ void uniform_fill_kernel(float* out, int count, UniformFill fill)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
    {
        out[index] = fill(index);
    }
}

// Writes a constant other than zero. cudaMemset can zero a buffer, not set another value.
__global__ void fill_constant_kernel(float* out, int count, float value)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
    {
        out[index] = value;
    }
}

// Launches the uniform-initialisation kernel. An empty tensor is skipped so the
// launch does not use a grid of zero blocks.
auto fill_uniform(dl::Tensor& tensor, float low, float high, unsigned long long seed) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    const int count = static_cast<int>(tensor.get_size());
    const dim3 grid(static_cast<unsigned int>((count + kFillThreads - 1) / kFillThreads));
    uniform_fill_kernel<<<grid, kFillThreads, 0, dl::current_stream()>>>(
        tensor.data(), count, UniformFill { low, high, seed });
    CHECK_CUDA(cudaGetLastError());
}

// Zeros the buffer with cudaMemsetAsync, and writes any other constant with the kernel.
// Gradients start at zero and do not need the kernel.
auto fill_constant(dl::Tensor& tensor, float value) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    if (value == 0.0F)
    {
        CHECK_CUDA(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), dl::current_stream()));
        CHECK_CUDA(cudaGetLastError());
        return;
    }
    const int count = static_cast<int>(tensor.get_size());
    const dim3 grid(static_cast<unsigned int>((count + kFillThreads - 1) / kFillThreads));
    fill_constant_kernel<<<grid, kFillThreads, 0, dl::current_stream()>>>(tensor.data(), count, value);
    CHECK_CUDA(cudaGetLastError());
}

// Creates or resizes the velocity buffer and zeros it when the shape changes.
// SGD momentum needs a velocity with the weight's shape; a fresh allocation left
// unzeroed would be garbage.
auto ensure_zero_like(std::optional<dl::Tensor>& slot, const dl::Tensor& like) -> dl::Tensor&
{
    if (!slot.has_value() || slot->get_shape() != like.get_shape() || slot->get_dtype() != like.get_dtype())
    {
        slot = dl::Tensor(like.get_shape(), like.get_device(), like.get_dtype());
        fill_constant(*slot, 0.0F);
    }
    return *slot;
}

// Rejects a host tensor or a null device pointer. The matrix product stays on the
// GPU and there is no host-to-device copy here.
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

// Copies weights device-to-device, converting on the current stream when the dtype
// differs. A loader may supply a type other than the layer buffer's.
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

// Requires a [batch, features] matrix with the given column count. The feature
// dimension is the contract of the product with the weight [input, output].
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

// Checks that the dimensions are positive and returns the weight shape [input, output].
// The error must fire before the constructor allocates the tensor.
auto fullyconnected_weight_shape(int input_size, int output_size) -> std::vector<int>
{
    if (input_size <= 0 || output_size <= 0)
    {
        throw std::runtime_error("FullyConnected requires positive input and output sizes");
    }
    return { input_size, output_size };
}

} // namespace

// Draws weights and bias from 1/sqrt(fan_in) and zeros the gradients. inertia_ is
// stored because backward uses it as the beta that accumulates dW, not as SGD momentum.
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
}

// Computes Y = X W + b and caches the input after a dtype conversion when needed.
// The output view comes from ensure, so the next forward overwrites the same buffer.
auto FullyConnected::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("FullyConnected_Forward");
    const dl::StreamGuard stream_guard(stream);
    require_rank2(input_tensor, input_size_, "FullyConnected::forward input");

    // matmul_into needs one dtype. A different input type is converted to the weight dtype.
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

// Accumulates dW and db and computes dX = dY W^T. inertia_ accumulates the parameter
// gradients; dX is overwritten because the input-gradient cache is not summed across calls.
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

    // inertia_ is the beta of matmul_into, so dW accumulates into the gradient buffer. It is not SGD momentum.
    input_cache_->matmul_into(*grad_output, weights_gradient_, true, false, inertia_);
    biases_gradient_.add_sum_rows_(*grad_output, inertia_);

    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, input_cache_->get_shape(), dl::Device::GPU,
        weights_.get_dtype());
    // beta 0 overwrites dX. The input-gradient cache is not summed across calls.
    grad_output->matmul_into(weights_, grad_input, false, true, 0.0F);
    input_cache_ready_ = false;
    return grad_input.as_view();
}

// Updates the weights with SGD, and when momentum > 0 keeps velocity in an optional
// buffer. A frozen layer returns immediately and does not touch the weights.
void FullyConnected::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("FullyConnected_Step");
    const dl::StreamGuard stream_guard(stream);
    if (frozen())
    {
        return;
    }
    const float clip = parameter_clip_bound();
    const float lr = step_learning_rate();
    if (momentum > 0.0F)
    {
        dl::Tensor& weight_velocity = ensure_zero_like(weights_velocity_, weights_);
        dl::Tensor& bias_velocity = ensure_zero_like(biases_velocity_, biases_);
        weights_.sgd_momentum_update_(weights_gradient_, weight_velocity, lr, momentum, weight_decay, clip);
        biases_.sgd_momentum_update_(biases_gradient_, bias_velocity, lr, momentum, weight_decay, clip);
        return;
    }
    weights_.sgd_update_(weights_gradient_, lr, weight_decay, clip);
    biases_.sgd_update_(biases_gradient_, lr, weight_decay, clip);
}

// Clips the weight and bias gradients to a symmetric range. A bound <= 0 turns clipping off.
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

// Returns views of the weights and bias for saving the network. Gradients and
// momentum velocity are not written to the file.
auto FullyConnected::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights", weights_.view(weights_.get_shape()));
    params.emplace("bias", biases_.view(biases_.get_shape()));
    return params;
}

// Loads the weights and bias with a device-to-device copy. The buffer shapes stay as they were at construction.
void FullyConnected::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    copy_same_size(weights_, params.at("weights"), "FullyConnected::set_parameters weights");
    copy_same_size(biases_, params.at("bias"), "FullyConnected::set_parameters bias");
}

// Leaves the parameters on the GPU. The matrix product has no host path.
auto FullyConnected::to(dl::Device device) -> void
{
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("FullyConnected parameters must remain on the GPU");
    }
    device_ = device;
}
