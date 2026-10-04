#include "DeepLearnLib/FusedCBR2d.hpp"
#include "DeepLearnLib/Nvtx.hpp"
#include "DeepLearnLib/SafeMath.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{

constexpr int kMomentThreads = 256;
constexpr int kElementwiseThreads = 256;
constexpr int kFillThreads = 256;

// Writes a constant into a GPU buffer. cudaMemset can only write zero, and gamma and the variance start at 1.
__global__ void fill_constant_kernel(float* out, int count, float value)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
    {
        out[index] = value;
    }
}

// Zeros the buffer with cudaMemsetAsync, and writes any other constant with the kernel.
// Batch-norm parameter initialisation cannot go through memset alone.
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
// SGD momentum on gamma and beta needs a velocity; a fresh allocation left unzeroed would be garbage.
auto ensure_zero_like(std::optional<dl::Tensor>& slot, const dl::Tensor& like) -> dl::Tensor&
{
    if (!slot.has_value() || slot->get_shape() != like.get_shape() || slot->get_dtype() != like.get_dtype())
    {
        slot = dl::Tensor(like.get_shape(), like.get_device(), like.get_dtype());
        fill_constant(*slot, 0.0F);
    }
    return *slot;
}

// Copies a parameter device-to-device, converting on the current stream when the
// dtype differs. Storing weights must not change the channel count.
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

// Rejects a tensor that is off the GPU or not rank 4. Convolution and batch-norm read NCHW layout.
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

// Loads an activation element as float. The moments and the BN+LeakyReLU fusion
// compute in float regardless of the storage type.
template <typename Act>
__device__ auto load_act(const Act* pointer, int index) -> float
{
    return pointer[index];
}

// Stores a float result into the activation buffer. One store serves both the fusion and the LeakyReLU derivative.
template <typename Act>
__device__ auto store_act(Act* pointer, int index, float value) -> void
{
    pointer[index] = value;
}

// One block per channel sums the values and the squares over the batch and the pixels,
// then reduces in shared memory. The batch mean and variance feed normalisation without global atomics.
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

// In train, computes the inverse standard deviation and folds the batch into running
// mean/var; in eval, substitutes the running statistics. A separate kernel, because the fusion and the cuDNN backward both read inv_std.
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
        // Batch-norm momentum, not SGD momentum: the new batch is mixed in with this weight.
        // The inverse std feeds the fusion and is kept for save_inv_var.
        const float batch_mean = mean[channel];
        const float batch_var = variance[channel];
        inv_std[channel] = rsqrtf(fmaxf(batch_var + epsilon, dl::kSafeEps));
        running_mean[channel] = ((1.0F - momentum) * running_mean[channel]) + (momentum * batch_mean);
        running_var[channel] = ((1.0F - momentum) * running_var[channel]) + (momentum * batch_var);
    }
    else
    {
        // Eval freezes running_*: they are substituted as this iteration's mean and scale.
        mean[channel] = running_mean[channel];
        inv_std[channel] = rsqrtf(fmaxf(running_var[channel] + epsilon, dl::kSafeEps));
    }
}

// In one pass, applies affine batch-norm and LeakyReLU. A separate kernel for the
// activation alone would read the whole tensor from global memory again.
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
    // LeakyReLU is fused onto the batch-norm output. Backward has to undo it before the BN backward.
    store_act(output, index, bn > 0.0F ? bn : bn * slope);
}

// Multiplies the gradient by 1 or by the slope according to the sign of the fused
// output. The pre-activation is not stored separately, so the LeakyReLU branch is recovered from the saved y.
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
    // Positive fused output has derivative 1; a non-positive one has derivative slope.
    store_act(grad_bn, index, incoming * (activated > 0.0F ? 1.0F : slope));
}

// Computes the block grid of the elementwise kernel. Rounding up by the element
// count and kElementwiseThreads covers the last, partial block.
auto elementwise_grid(int count) -> dim3
{
    return dim3(static_cast<unsigned int>((count + kElementwiseThreads - 1) / kElementwiseThreads));
}

// Returns the batch-norm parameter shape [1, C, 1, 1]. That layout matches spatial
// batch-norm and the gamma, beta, and per-channel statistic buffers.
auto channel_shape(int channels) -> std::vector<int>
{
    return { 1, channels, 1, 1 };
}

} // namespace

// Builds the convolution together with the batch-norm buffers. Gamma starts at 1
// and the running variance starts at 1, so the first evaluation has a finite normalisation.
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

// Turns training on for this layer and for the inner convolution. Otherwise Conv2d
// would stay in eval and would not keep the cache backward needs.
void FusedCBR2d::train()
{
    Layer::train();
    conv_.train();
}

// Switches the layer and the convolution to inference. Batch-norm then reads the
// running stats, and the convolution itself uses the same math as in train.
void FusedCBR2d::eval()
{
    Layer::eval();
    conv_.eval();
}

// Rebuilds the NCHW and spatial BN descriptors only when the convolution output
// shape changes. Derive keeps the gamma layout consistent with CUDNN_BATCHNORM_SPATIAL.
auto FusedCBR2d::configure_bn_descriptors(const dl::Tensor& conv_output) -> void
{
    const auto& shape = conv_output.get_shape();
    if (bn_descriptors_configured_ && shape == bn_shape_cache_)
    {
        return;
    }
    x_desc_.set_nchw(shape[0], shape[1], shape[2], shape[3], cudnn_data_type(conv_output.get_dtype()));
    // Derived from the convolution output so gamma matches CUDNN_BATCHNORM_SPATIAL.
    CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(bn_desc_.get(), x_desc_.get(), CUDNN_BATCHNORM_SPATIAL));
    bn_shape_cache_ = shape;
    bn_descriptors_configured_ = true;
}

// In train, computes the batch moments, then in both modes finishes the statistics
// and launches the BN+LeakyReLU fusion. An empty tensor or a zero spatial plane returns early, because the reduction would divide by zero.
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
        // Empty tensor or a zero spatial plane: the reduction would divide by zero.
        return;
    }

    if (is_training_)
    {
        spatial_moments_kernel<float><<<static_cast<unsigned int>(channels), kMomentThreads, 0, stream>>>(
            conv_output.data(), save_mean_.data(), batch_var_.data(), batch, channels, spatial);
        CHECK_CUDA(cudaGetLastError());
    }

    finalize_bn_stats_kernel<<<elementwise_grid(channels), kElementwiseThreads, 0, stream>>>(save_mean_.data(),
        batch_var_.data(), save_inv_var_.data(), running_mean_.data(), running_var_.data(), channels, epsilon,
        bn_momentum_, is_training_);
    CHECK_CUDA(cudaGetLastError());

    fused_bn_leaky_kernel<float><<<elementwise_grid(total), kElementwiseThreads, 0, stream>>>(conv_output.data(),
        output.data(), save_mean_.data(), save_inv_var_.data(), gamma_.data(), beta_.data(), leaky_slope_, total,
        channels, spatial);
    CHECK_CUDA(cudaGetLastError());
}

// Runs convolution-with-bias, then the BN and LeakyReLU fusion into the cache. In
// train it keeps the convolution output, because the batch-norm backward needs x from before normalisation.
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
        // Pre-normalisation activation. The batch-norm backward reads it as x.
        bn_input_cache_ = conv_output.as_view();
    }

    dl::Tensor& fused = dl::Tensor::ensure(fused_output_cache_, conv_output.get_shape(), dl::Device::GPU,
        conv_output.get_dtype());
    apply_bn_leaky_into(conv_output, fused, stream);
    caches_ready_ = is_training_;
    return fused.as_view();
}

// LeakyReLU derivative first, then cudnnBatchNormalizationBackward, then the convolution
// backward. The gradient that reaches the convolution is already past the activation and batch-norm, so Conv2d does not see the raw loss.
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
        // Forward fused conv, then batch-norm, then LeakyReLU. Undo LeakyReLU before the batch-norm backward.
        leaky_backward_from_output_kernel<float><<<elementwise_grid(total), kElementwiseThreads, 0, stream>>>(
            fused_output_cache_->data(), grad_output->data(), grad_bn.data(), leaky_slope_, total);
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

// Copies the optimiser hyperparameters onto the inner convolution and takes an SGD
// step on gamma and beta. Without the learning_rate copy, the convolution step would keep the inner layer's previous step.
void FusedCBR2d::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("FusedCBR2d_Step");
    if (frozen())
    {
        return;
    }
    // The convolution keeps its own optimiser fields. Its step uses the same values as batch-norm.
    conv_.learning_rate = learning_rate;
    conv_.gradient_clip = gradient_clip;
    conv_.momentum = momentum;
    conv_.weight_decay = weight_decay;
    conv_.step(stream);
    const float clip = parameter_clip_bound();
    const float lr = step_learning_rate();
    if (momentum > 0.0F)
    {
        dl::Tensor& gamma_velocity = ensure_zero_like(gamma_velocity_, gamma_);
        dl::Tensor& beta_velocity = ensure_zero_like(beta_velocity_, beta_);
        gamma_.sgd_momentum_update_(gamma_grad_, gamma_velocity, lr, momentum, weight_decay, clip);
        beta_.sgd_momentum_update_(beta_grad_, beta_velocity, lr, momentum, weight_decay, clip);
        return;
    }
    gamma_.sgd_update_(gamma_grad_, lr, weight_decay, clip);
    beta_.sgd_update_(beta_grad_, lr, weight_decay, clip);
}

// Clips the convolution gradients and gamma and beta. A bound <= 0 is a no-op,
// as in the base clip_gradients.
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

// Appends gamma, beta, and the running stats to the convolution parameter map. One
// network save covers both the convolution weights and the normalisation.
auto FusedCBR2d::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params = conv_.get_parameters();
    params.emplace("gamma", gamma_.view(gamma_.get_shape()));
    params.emplace("beta", beta_.view(beta_.get_shape()));
    params.emplace("running_mean", running_mean_.view(running_mean_.get_shape()));
    params.emplace("running_var", running_var_.view(running_var_.get_shape()));
    return params;
}

// Loads the convolution parameters and the four batch-norm tensors. The copy checks
// the size so a weight file cannot change the channel count.
void FusedCBR2d::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    conv_.set_parameters(params);
    copy_same_size(gamma_, params.at("gamma"), "FusedCBR2d::set_parameters gamma");
    copy_same_size(beta_, params.at("beta"), "FusedCBR2d::set_parameters beta");
    copy_same_size(running_mean_, params.at("running_mean"), "FusedCBR2d::set_parameters running_mean");
    copy_same_size(running_var_, params.at("running_var"), "FusedCBR2d::set_parameters running_var");
}

// Forwards the device to the convolution and stays on the GPU. The batch-norm buffers have no host copy.
auto FusedCBR2d::to(dl::Device device) -> void
{
    conv_.to(device);
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("FusedCBR2d parameters must remain on the GPU");
    }
    device_ = device;
}
