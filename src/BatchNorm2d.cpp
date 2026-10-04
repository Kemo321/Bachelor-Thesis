#include "DeepLearnLib/BatchNorm2d.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace
{

constexpr int kFillThreads = 256;

// Writes a constant into a GPU buffer. A separate kernel, because cudaMemset can only write zero.
__global__ void fill_constant_kernel(float* out, int count, float value)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
    {
        out[index] = value;
    }
}

// Zeros the buffer with cudaMemsetAsync, and writes any other constant with the kernel.
// Gamma and the running variance start at 1, so memset alone is not enough.
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

// Rejects a tensor that is off the GPU or not rank 4. Spatial cuDNN batch-norm
// reads NCHW layout from the descriptor.
auto require_gpu_nchw(const dl::Tensor& tensor, const char* name) -> void
{
    if (tensor.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " must reside on the GPU");
    }
    if (tensor.get_shape().size() != 4)
    {
        throw std::runtime_error(std::string(name) + " must have NCHW rank 4");
    }
    if (tensor.get_size() > 0 && tensor.data() == nullptr)
    {
        throw std::runtime_error(std::string(name) + " has a null device pointer");
    }
}

// Copies a parameter device-to-device when the sizes match. Loading weights must
// not change the shape of the layer buffers.
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
    dl::memcpy_d2d_on_current(dst.data(), src.data(), src.nbytes());
}

// Returns the shape [1, C, 1, 1] and checks the channel count and eps. That layout
// matches the cuDNN spatial batch-norm descriptor.
auto batchnorm_channel_shape(int num_features, float eps) -> std::vector<int>
{
    if (num_features <= 0)
    {
        throw std::runtime_error("BatchNorm2d requires a positive channel count");
    }
    if (eps < 0.0F)
    {
        throw std::runtime_error("BatchNorm2d epsilon must be non-negative");
    }
    return { 1, num_features, 1, 1 };
}

} // namespace

// Allocates gamma, beta, the running statistics, and the save buffers on the GPU.
// Gamma starts at 1 and the running variance starts at 1, so the first inference does not divide by zero.
BatchNorm2d::BatchNorm2d(int num_features, float eps, float momentum)
    : num_features_(num_features)
    , eps_(eps)
    , momentum_bn_(momentum)
    , gamma_(batchnorm_channel_shape(num_features, eps), dl::Device::GPU)
    , beta_({ 1, num_features, 1, 1 }, dl::Device::GPU)
    , gamma_grad_({ 1, num_features, 1, 1 }, dl::Device::GPU)
    , beta_grad_({ 1, num_features, 1, 1 }, dl::Device::GPU)
    , running_mean_({ 1, num_features, 1, 1 }, dl::Device::GPU)
    , running_var_({ 1, num_features, 1, 1 }, dl::Device::GPU)
    , save_mean_({ 1, num_features, 1, 1 }, dl::Device::GPU)
    , save_inv_var_({ 1, num_features, 1, 1 }, dl::Device::GPU)
{
    device_ = dl::Device::GPU;
    fill_constant(gamma_, 1.0F);
    fill_constant(beta_, 0.0F);
    fill_constant(gamma_grad_, 0.0F);
    fill_constant(beta_grad_, 0.0F);
    fill_constant(running_mean_, 0.0F);
    fill_constant(running_var_, 1.0F);
}

// Sets the NCHW descriptor and derives the spatial BN descriptor from it. Derive,
// rather than a hand-written shape, keeps the gamma scale consistent with CUDNN_BATCHNORM_SPATIAL.
auto BatchNorm2d::configure_descriptors(int batch, int channels, int height, int width, dl::Dtype dtype) -> void
{
    const std::vector<int> shape { batch, channels, height, width };
    if (channels != num_features_)
    {
        throw std::runtime_error("BatchNorm2d channel count does not match the layer");
    }

    x_desc_.set_nchw(batch, channels, height, width, cudnn_data_type(dtype));
    // Derived from x, so the per-channel scale matches CUDNN_BATCHNORM_SPATIAL.
    CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(bn_desc_.get(), x_desc_.get(), CUDNN_BATCHNORM_SPATIAL));
    input_shape_cache_ = shape;
    descriptors_configured_ = true;
}

// In train, computes batch statistics and updates running mean/var; in eval, normalises
// with the saved statistics. The input is cached only in train, because backward reads x and save_*.
auto BatchNorm2d::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("BatchNorm2d_Forward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    require_gpu_nchw(input_tensor, "BatchNorm2d::forward input");

    const int batch = input_tensor.get_shape()[0];
    const int channels = input_tensor.get_shape()[1];
    const int height = input_tensor.get_shape()[2];
    const int width = input_tensor.get_shape()[3];
    configure_descriptors(batch, channels, height, width, input_tensor.get_dtype());

    dl::Tensor& output = dl::Tensor::ensure(output_cache_, input_tensor.get_shape(), dl::Device::GPU,
        input_tensor.get_dtype());
    const float alpha { 1.0F };
    const float beta_zero { 0.0F };
    const auto handle = dl::get_cudnn_handle();
    // cuDNN rejects an epsilon below its minimum.
    const double epsilon = std::max(static_cast<double>(eps_), static_cast<double>(CUDNN_BN_MIN_EPSILON));

    if (is_training_)
    {
        input_cache_ = input_tensor.as_view();
        input_cache_ready_ = true;

        // Train folds this batch into running_mean and running_var.
        // momentum_bn_ is the weight of the new batch in that cuDNN average. It is not SGD momentum,
        // and it is not the factor that keeps the previous running value.
        const double average_factor = static_cast<double>(momentum_bn_);
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_zero, x_desc_.get(), input_tensor.data(), x_desc_.get(),
            output.data(), bn_desc_.get(), gamma_.data(), beta_.data(), average_factor, running_mean_.data(),
            running_var_.data(), epsilon, save_mean_.data(), save_inv_var_.data()));
    }
    else
    {
        // Eval freezes running_mean and running_var and normalises with them. No batch stats, no backward cache.
        input_cache_ready_ = false;
        CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
            handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_zero, x_desc_.get(), input_tensor.data(), x_desc_.get(),
            output.data(), bn_desc_.get(), gamma_.data(), beta_.data(), running_mean_.data(), running_var_.data(),
            epsilon));
    }

    return output.as_view();
}

// Computes dx, dgamma, and dbeta from the statistics saved by the training forward.
// In eval, backward has neither the input nor save_mean / save_inv_var.
auto BatchNorm2d::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("BatchNorm2d_Backward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    if (!is_training_ || !input_cache_ready_ || !input_cache_.has_value())
    {
        throw std::runtime_error("BatchNorm2d::backward requires a preceding training forward pass");
    }
    require_gpu_nchw(output_error_derivative, "BatchNorm2d::backward grad_output");
    if (output_error_derivative.get_shape() != input_cache_->get_shape())
    {
        throw std::runtime_error("BatchNorm2d::backward grad_output shape does not match the cached input");
    }

    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, input_cache_->get_shape(), dl::Device::GPU,
        input_cache_->get_dtype());
    const float alpha { 1.0F };
    const float beta_zero { 0.0F };
    const double epsilon = std::max(static_cast<double>(eps_), static_cast<double>(CUDNN_BN_MIN_EPSILON));

    CHECK_CUDNN(cudnnBatchNormalizationBackward(
        dl::get_cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_zero, &alpha, &beta_zero, x_desc_.get(),
        input_cache_->data(), x_desc_.get(), output_error_derivative.data(), x_desc_.get(), grad_input.data(),
        bn_desc_.get(), gamma_.data(), gamma_grad_.data(), beta_grad_.data(), epsilon, save_mean_.data(),
        save_inv_var_.data()));

    input_cache_ready_ = false;
    return grad_input.as_view();
}

// Takes an SGD step on gamma and beta. Running mean and variance are not touched
// here: cuDNN already writes them in the training forward.
void BatchNorm2d::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("BatchNorm2d_Step");
    const dl::StreamGuard stream_guard(stream);
    gamma_.sgd_update_(gamma_grad_, step_learning_rate(), weight_decay, parameter_clip_bound());
    beta_.sgd_update_(beta_grad_, step_learning_rate(), weight_decay, parameter_clip_bound());
}

// Clips the gamma and beta gradients to a symmetric range. A bound <= 0 turns
// clipping off, as parameter_clip_bound does.
void BatchNorm2d::clip_gradients(float abs_bound, cudaStream_t stream)
{
    const dl::StreamGuard stream_guard(stream);
    if (abs_bound <= 0.0F)
    {
        return;
    }
    gamma_grad_.clamp_(-abs_bound, abs_bound);
    beta_grad_.clamp_(-abs_bound, abs_bound);
}

// Returns views of gamma, beta, and the running statistics for saving the network.
// Gradients and the save buffers are not written to the weight file.
auto BatchNorm2d::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params;
    params.emplace("gamma", gamma_.view(gamma_.get_shape()));
    params.emplace("beta", beta_.view(beta_.get_shape()));
    params.emplace("running_mean", running_mean_.view(running_mean_.get_shape()));
    params.emplace("running_var", running_var_.view(running_var_.get_shape()));
    return params;
}

// Loads the saved tensors into the layer buffers with a device-to-device copy.
// The shape stays as it was at construction; only the contents are copied.
void BatchNorm2d::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    copy_same_size(gamma_, params.at("gamma"), "BatchNorm2d::set_parameters gamma");
    copy_same_size(beta_, params.at("beta"), "BatchNorm2d::set_parameters beta");
    copy_same_size(running_mean_, params.at("running_mean"), "BatchNorm2d::set_parameters running_mean");
    copy_same_size(running_var_, params.at("running_var"), "BatchNorm2d::set_parameters running_var");
}

// Leaves the parameters on the GPU. The layer has no host path for the cuDNN calls.
auto BatchNorm2d::to(dl::Device device) -> void
{
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("BatchNorm2d parameters must remain on the GPU");
    }
    device_ = device;
}
