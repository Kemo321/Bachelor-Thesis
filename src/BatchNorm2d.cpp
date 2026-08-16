#include "DeepLearnLib/BatchNorm2d.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace
{

constexpr float kWeightDecay = 0.0005F;

struct ScaledAdd
{
    float scale;

    __host__ __device__ auto operator()(float lhs, float rhs) const -> float
    {
        return lhs + (scale * rhs);
    }
};

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

auto add_scaled(dl::Tensor& lhs, const dl::Tensor& rhs, float scale) -> void
{
    if (lhs.get_size() == 0)
    {
        return;
    }
    auto dest = thrust::device_pointer_cast(lhs.data());
    auto src = thrust::device_pointer_cast(rhs.data());
    thrust::transform(thrust::cuda::par.on(dl::current_stream()), dest, dest + static_cast<std::ptrdiff_t>(lhs.get_size()), src, dest,
        ScaledAdd { scale });
    CHECK_CUDA(cudaGetLastError());
}

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

auto BatchNorm2d::configure_descriptors(int batch, int channels, int height, int width, dl::Dtype dtype) -> void
{
    const std::vector<int> shape { batch, channels, height, width };
    if (channels != num_features_)
    {
        throw std::runtime_error("BatchNorm2d channel count does not match the layer");
    }

    x_desc_.set_nchw(batch, channels, height, width, cudnn_data_type(dtype));
    CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(bn_desc_.get(), x_desc_.get(), CUDNN_BATCHNORM_SPATIAL));
    input_shape_cache_ = shape;
    descriptors_configured_ = true;
}

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

    dl::Tensor output(input_tensor.get_shape(), dl::Device::GPU, input_tensor.get_dtype());
    const float alpha { 1.0F };
    const float beta_zero { 0.0F };
    const auto handle = dl::get_cudnn_handle();
    const double epsilon = std::max(static_cast<double>(eps_), static_cast<double>(CUDNN_BN_MIN_EPSILON));

    if (is_training_)
    {
        input_cache_ = dl::Tensor(input_tensor.get_shape(), dl::Device::GPU, input_tensor.get_dtype());
        copy_same_size(*input_cache_, input_tensor, "BatchNorm2d::forward input cache");

        const double average_factor = static_cast<double>(momentum_bn_);
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_zero, x_desc_.get(), input_tensor.data(), x_desc_.get(),
            output.data(), bn_desc_.get(), gamma_.data(), beta_.data(), average_factor, running_mean_.data(),
            running_var_.data(), epsilon, save_mean_.data(), save_inv_var_.data()));
    }
    else
    {
        input_cache_.reset();
        CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
            handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_zero, x_desc_.get(), input_tensor.data(), x_desc_.get(),
            output.data(), bn_desc_.get(), gamma_.data(), beta_.data(), running_mean_.data(), running_var_.data(),
            epsilon));
    }

    return output;
}

auto BatchNorm2d::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("BatchNorm2d_Backward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    if (!is_training_ || !input_cache_.has_value())
    {
        throw std::runtime_error("BatchNorm2d::backward requires a preceding training forward pass");
    }
    require_gpu_nchw(output_error_derivative, "BatchNorm2d::backward grad_output");
    if (output_error_derivative.get_shape() != input_cache_->get_shape())
    {
        throw std::runtime_error("BatchNorm2d::backward grad_output shape does not match the cached input");
    }

    dl::Tensor grad_input(input_cache_->get_shape(), dl::Device::GPU, input_cache_->get_dtype());
    const float alpha { 1.0F };
    const float beta_zero { 0.0F };
    const double epsilon = std::max(static_cast<double>(eps_), static_cast<double>(CUDNN_BN_MIN_EPSILON));

    CHECK_CUDNN(cudnnBatchNormalizationBackward(
        dl::get_cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_zero, &alpha, &beta_zero, x_desc_.get(),
        input_cache_->data(), x_desc_.get(), output_error_derivative.data(), x_desc_.get(), grad_input.data(),
        bn_desc_.get(), gamma_.data(), gamma_grad_.data(), beta_grad_.data(), epsilon, save_mean_.data(),
        save_inv_var_.data()));

    add_scaled(gamma_grad_, gamma_, kWeightDecay);
    add_scaled(beta_grad_, beta_, kWeightDecay);
    input_cache_.reset();
    return grad_input;
}

void BatchNorm2d::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("BatchNorm2d_Step");
    const dl::StreamGuard stream_guard(stream);
    gamma_ = gamma_ - (gamma_grad_ * scaled_learning_rate());
    beta_ = beta_ - (beta_grad_ * scaled_learning_rate());
}

void BatchNorm2d::clip_gradients(float abs_bound, cudaStream_t stream)
{
    const dl::StreamGuard stream_guard(stream);
    if (abs_bound <= 0.0F)
    {
        return;
    }
    gamma_grad_ = gamma_grad_.clamp(-abs_bound, abs_bound);
    beta_grad_ = beta_grad_.clamp(-abs_bound, abs_bound);
}

auto BatchNorm2d::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params;
    params.emplace("gamma", gamma_.view(gamma_.get_shape()));
    params.emplace("beta", beta_.view(beta_.get_shape()));
    params.emplace("running_mean", running_mean_.view(running_mean_.get_shape()));
    params.emplace("running_var", running_var_.view(running_var_.get_shape()));
    return params;
}

void BatchNorm2d::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    copy_same_size(gamma_, params.at("gamma"), "BatchNorm2d::set_parameters gamma");
    copy_same_size(beta_, params.at("beta"), "BatchNorm2d::set_parameters beta");
    copy_same_size(running_mean_, params.at("running_mean"), "BatchNorm2d::set_parameters running_mean");
    copy_same_size(running_var_, params.at("running_var"), "BatchNorm2d::set_parameters running_var");
}

auto BatchNorm2d::to(dl::Device device) -> void
{
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("BatchNorm2d parameters must remain on the GPU");
    }
    device_ = device;
}
