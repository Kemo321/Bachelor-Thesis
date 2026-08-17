#include "DeepLearnLib/FusedCBR2d.hpp"
#include "DeepLearnLib/Nvtx.hpp"
#include "DeepLearnLib/SafeMath.hpp"

#include <algorithm>
#include <cstddef>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace
{

constexpr float kWeightDecay = 0.0005F;
constexpr int kMomentThreads = 256;
constexpr int kElementwiseThreads = 256;

auto fill_constant(dl::Tensor& tensor, float value) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    auto out = thrust::device_pointer_cast(tensor.data());
    thrust::fill(thrust::cuda::par.on(dl::current_stream()), out, out + static_cast<std::ptrdiff_t>(tensor.get_size()),
        value);
    CHECK_CUDA(cudaGetLastError());
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
__global__ void spatial_moments_kernel(const Act* input, float* mean, float* variance, int batch, int channels,
    int spatial)
{
    const int channel = static_cast<int>(blockIdx.x);
    if (channel >= channels)
    {
        return;
    }

    const int count = batch * spatial;
    float sum = 0.0F;
    float sum_sq = 0.0F;
    for (int item = static_cast<int>(threadIdx.x); item < count; item += static_cast<int>(blockDim.x))
    {
        const int sample = item / spatial;
        const int inner = item % spatial;
        const int index = ((sample * channels + channel) * spatial) + inner;
        const float value = load_act(input, index);
        sum += value;
        sum_sq += value * value;
    }

    __shared__ float shared_sum[kMomentThreads];
    __shared__ float shared_sum_sq[kMomentThreads];
    shared_sum[threadIdx.x] = sum;
    shared_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1)
    {
        if (static_cast<int>(threadIdx.x) < stride)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        const float inv_count = 1.0F / static_cast<float>(count);
        const float channel_mean = shared_sum[0] * inv_count;
        const float channel_var = fmaxf((shared_sum_sq[0] * inv_count) - (channel_mean * channel_mean), 0.0F);
        mean[channel] = channel_mean;
        variance[channel] = channel_var;
    }
}

__global__ void finalize_bn_stats_kernel(float* mean, float* variance, float* inv_std, float* running_mean,
    float* running_var, int channels, float epsilon, float momentum, bool training)
{
    const int channel = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (channel >= channels)
    {
        return;
    }

    if (training)
    {
        const float batch_mean = mean[channel];
        const float batch_var = variance[channel];
        inv_std[channel] = rsqrtf(fmaxf(batch_var + epsilon, dl::kSafeEps));
        running_mean[channel] = ((1.0F - momentum) * running_mean[channel]) + (momentum * batch_mean);
        running_var[channel] = ((1.0F - momentum) * running_var[channel]) + (momentum * batch_var);
    }
    else
    {
        mean[channel] = running_mean[channel];
        inv_std[channel] = rsqrtf(fmaxf(running_var[channel] + epsilon, dl::kSafeEps));
    }
}

template <typename Act>
__global__ void fused_bn_leaky_kernel(const Act* input, Act* output, const float* mean, const float* inv_std,
    const float* gamma, const float* beta, float slope, int total, int channels, int spatial)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= total)
    {
        return;
    }
    const int channel = (index / spatial) % channels;
    const float normalized = (load_act(input, index) - mean[channel]) * inv_std[channel];
    const float bn = (gamma[channel] * normalized) + beta[channel];
    store_act(output, index, bn > 0.0F ? bn : bn * slope);
}

template <typename Act>
__global__ void leaky_backward_from_output_kernel(const Act* fused_output, const Act* grad_output, Act* grad_bn,
    float slope, int total)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= total)
    {
        return;
    }
    const float activated = load_act(fused_output, index);
    const float incoming = load_act(grad_output, index);
    store_act(grad_bn, index, incoming * (activated > 0.0F ? 1.0F : slope));
}

auto elementwise_grid(int count) -> dim3
{
    return dim3(static_cast<unsigned int>((count + kElementwiseThreads - 1) / kElementwiseThreads));
}

auto channel_shape(int channels) -> std::vector<int>
{
    return { 1, channels, 1, 1 };
}

} // namespace

FusedCBR2d::FusedCBR2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val,
    float leaky_slope, float bn_eps, float bn_momentum)
    : conv_(in_channels, out_channels, kernel_size, stride_val, padding_val)
    , leaky_slope_(leaky_slope)
    , bn_eps_(bn_eps)
    , bn_momentum_(bn_momentum)
    , out_channels_(out_channels)
    , gamma_(channel_shape(out_channels), dl::Device::GPU)
    , beta_(channel_shape(out_channels), dl::Device::GPU)
    , gamma_grad_(channel_shape(out_channels), dl::Device::GPU)
    , beta_grad_(channel_shape(out_channels), dl::Device::GPU)
    , running_mean_(channel_shape(out_channels), dl::Device::GPU)
    , running_var_(channel_shape(out_channels), dl::Device::GPU)
    , batch_var_(channel_shape(out_channels), dl::Device::GPU)
    , save_mean_(channel_shape(out_channels), dl::Device::GPU)
    , save_inv_var_(channel_shape(out_channels), dl::Device::GPU)
{
    if (bn_eps < 0.0F)
    {
        throw std::runtime_error("FusedCBR2d epsilon must be non-negative");
    }
    if (bn_momentum < 0.0F || bn_momentum > 1.0F)
    {
        throw std::runtime_error("FusedCBR2d BatchNorm momentum must be in [0, 1]");
    }

    device_ = dl::Device::GPU;
    fill_constant(gamma_, 1.0F);
    fill_constant(beta_, 0.0F);
    fill_constant(gamma_grad_, 0.0F);
    fill_constant(beta_grad_, 0.0F);
    fill_constant(running_mean_, 0.0F);
    fill_constant(running_var_, 1.0F);
    fill_constant(batch_var_, 1.0F);
    fill_constant(save_mean_, 0.0F);
    fill_constant(save_inv_var_, 1.0F);
}

void FusedCBR2d::train()
{
    Layer::train();
    conv_.train();
}

void FusedCBR2d::eval()
{
    Layer::eval();
    conv_.eval();
}

auto FusedCBR2d::configure_bn_descriptors(const dl::Tensor& conv_output) -> void
{
    const auto& shape = conv_output.get_shape();
    if (bn_descriptors_configured_ && shape == bn_shape_cache_)
    {
        return;
    }
    x_desc_.set_nchw(shape[0], shape[1], shape[2], shape[3], cudnn_data_type(conv_output.get_dtype()));
    CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(bn_desc_.get(), x_desc_.get(), CUDNN_BATCHNORM_SPATIAL));
    bn_shape_cache_ = shape;
    bn_descriptors_configured_ = true;
}

auto FusedCBR2d::apply_bn_leaky_into(const dl::Tensor& conv_output, dl::Tensor& output, cudaStream_t stream) -> void
{
    const int batch = conv_output.get_shape()[0];
    const int channels = conv_output.get_shape()[1];
    if (channels != out_channels_)
    {
        throw std::runtime_error("FusedCBR2d BatchNorm channel count does not match the convolution");
    }
    const int height = conv_output.get_shape()[2];
    const int width = conv_output.get_shape()[3];
    const int spatial = height * width;
    const int total = static_cast<int>(conv_output.get_size());
    const float epsilon = std::max(bn_eps_, static_cast<float>(CUDNN_BN_MIN_EPSILON));

    if (total == 0 || spatial == 0)
    {
        return;
    }

    if (is_training_)
    {
        if (conv_output.get_dtype() == dl::Dtype::Float16)
        {
            spatial_moments_kernel<__half><<<static_cast<unsigned int>(channels), kMomentThreads, 0, stream>>>(
                conv_output.half_data(), save_mean_.data(), batch_var_.data(), batch, channels, spatial);
        }
        else
        {
            spatial_moments_kernel<float><<<static_cast<unsigned int>(channels), kMomentThreads, 0, stream>>>(
                conv_output.data(), save_mean_.data(), batch_var_.data(), batch, channels, spatial);
        }
        CHECK_CUDA(cudaGetLastError());
    }

    finalize_bn_stats_kernel<<<elementwise_grid(channels), kElementwiseThreads, 0, stream>>>(save_mean_.data(),
        batch_var_.data(), save_inv_var_.data(), running_mean_.data(), running_var_.data(), channels, epsilon,
        bn_momentum_, is_training_);
    CHECK_CUDA(cudaGetLastError());

    if (conv_output.get_dtype() == dl::Dtype::Float16)
    {
        fused_bn_leaky_kernel<__half><<<elementwise_grid(total), kElementwiseThreads, 0, stream>>>(
            conv_output.half_data(), output.half_data(), save_mean_.data(), save_inv_var_.data(), gamma_.data(),
            beta_.data(), leaky_slope_, total, channels, spatial);
    }
    else
    {
        fused_bn_leaky_kernel<float><<<elementwise_grid(total), kElementwiseThreads, 0, stream>>>(conv_output.data(),
            output.data(), save_mean_.data(), save_inv_var_.data(), gamma_.data(), beta_.data(), leaky_slope_, total,
            channels, spatial);
    }
    CHECK_CUDA(cudaGetLastError());
}

auto FusedCBR2d::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("FusedCBR2d_Forward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    require_gpu_nchw(input_tensor, "FusedCBR2d::forward input");

    dl::Tensor conv_output = conv_.forward(input_tensor, stream);
    configure_bn_descriptors(conv_output);

    if (is_training_)
    {
        bn_input_cache_ = conv_output.as_view();
    }

    dl::Tensor& fused = dl::Tensor::ensure(fused_output_cache_, conv_output.get_shape(), dl::Device::GPU,
        conv_output.get_dtype());
    apply_bn_leaky_into(conv_output, fused, stream);
    caches_ready_ = is_training_;
    return fused.as_view();
}

auto FusedCBR2d::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("FusedCBR2d_Backward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    if (!is_training_ || !caches_ready_ || !bn_input_cache_.has_value() || !fused_output_cache_.has_value())
    {
        throw std::runtime_error("FusedCBR2d::backward requires a preceding training forward pass");
    }
    require_gpu_nchw(output_error_derivative, "FusedCBR2d::backward grad_output");
    if (output_error_derivative.get_shape() != fused_output_cache_->get_shape())
    {
        throw std::runtime_error("FusedCBR2d::backward grad_output shape does not match the fused forward output");
    }

    const int total = static_cast<int>(output_error_derivative.get_size());
    const dl::Tensor* grad_output = &output_error_derivative;
    dl::Tensor converted_grad;
    if (output_error_derivative.get_dtype() != fused_output_cache_->get_dtype())
    {
        converted_grad = output_error_derivative.to_dtype(fused_output_cache_->get_dtype(), stream);
        grad_output = &converted_grad;
    }

    dl::Tensor& grad_bn = dl::Tensor::ensure(grad_bn_cache_, fused_output_cache_->get_shape(), dl::Device::GPU,
        fused_output_cache_->get_dtype());
    if (total > 0)
    {
        if (fused_output_cache_->get_dtype() == dl::Dtype::Float16)
        {
            leaky_backward_from_output_kernel<__half><<<elementwise_grid(total), kElementwiseThreads, 0, stream>>>(
                fused_output_cache_->half_data(), grad_output->half_data(), grad_bn.half_data(), leaky_slope_, total);
        }
        else
        {
            leaky_backward_from_output_kernel<float><<<elementwise_grid(total), kElementwiseThreads, 0, stream>>>(
                fused_output_cache_->data(), grad_output->data(), grad_bn.data(), leaky_slope_, total);
        }
        CHECK_CUDA(cudaGetLastError());
    }

    dl::Tensor& grad_conv = dl::Tensor::ensure(grad_conv_cache_, bn_input_cache_->get_shape(), dl::Device::GPU,
        bn_input_cache_->get_dtype());
    const float alpha { 1.0F };
    const float beta_zero { 0.0F };
    const double epsilon = std::max(static_cast<double>(bn_eps_), static_cast<double>(CUDNN_BN_MIN_EPSILON));
    CHECK_CUDNN(cudnnBatchNormalizationBackward(dl::get_cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta_zero,
        &alpha, &beta_zero, x_desc_.get(), bn_input_cache_->data(), x_desc_.get(), grad_bn.data(), x_desc_.get(),
        grad_conv.data(), bn_desc_.get(), gamma_.data(), gamma_grad_.data(), beta_grad_.data(), epsilon,
        save_mean_.data(), save_inv_var_.data()));

    caches_ready_ = false;
    return conv_.backward(grad_conv, stream);
}

void FusedCBR2d::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("FusedCBR2d_Step");
    conv_.learning_rate = learning_rate;
    conv_.gradient_clip = gradient_clip;
    conv_.step(stream);
    gamma_.sgd_update_(gamma_grad_, scaled_learning_rate(), kWeightDecay, parameter_clip_bound());
    beta_.sgd_update_(beta_grad_, scaled_learning_rate(), kWeightDecay, parameter_clip_bound());
}

void FusedCBR2d::clip_gradients(float abs_bound, cudaStream_t stream)
{
    if (abs_bound <= 0.0F)
    {
        return;
    }
    conv_.clip_gradients(abs_bound, stream);
    gamma_grad_.clamp_(-abs_bound, abs_bound);
    beta_grad_.clamp_(-abs_bound, abs_bound);
}

auto FusedCBR2d::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params = conv_.get_parameters();
    params.emplace("gamma", gamma_.view(gamma_.get_shape()));
    params.emplace("beta", beta_.view(beta_.get_shape()));
    params.emplace("running_mean", running_mean_.view(running_mean_.get_shape()));
    params.emplace("running_var", running_var_.view(running_var_.get_shape()));
    return params;
}

void FusedCBR2d::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    conv_.set_parameters(params);
    copy_same_size(gamma_, params.at("gamma"), "FusedCBR2d::set_parameters gamma");
    copy_same_size(beta_, params.at("beta"), "FusedCBR2d::set_parameters beta");
    copy_same_size(running_mean_, params.at("running_mean"), "FusedCBR2d::set_parameters running_mean");
    copy_same_size(running_var_, params.at("running_var"), "FusedCBR2d::set_parameters running_var");
}

auto FusedCBR2d::to(dl::Device device) -> void
{
    conv_.to(device);
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("FusedCBR2d parameters must remain on the GPU");
    }
    device_ = device;
}
