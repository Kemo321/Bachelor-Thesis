#include "DeepLearnLib/LocalLayer.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr float kWeightDecay = 0.0005F;
constexpr int kFillThreads = 256;
constexpr int kLocalThreads = 256;

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

__global__ void uniform_fill_kernel(float* out, int count, UniformFill fill)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
    {
        out[index] = fill(index);
    }
}

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

auto fill_zero(dl::Tensor& tensor) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    CHECK_CUDA(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), dl::current_stream()));
    CHECK_CUDA(cudaGetLastError());
}

auto ensure_zero_like(std::optional<dl::Tensor>& slot, const dl::Tensor& like) -> dl::Tensor&
{
    if (!slot.has_value() || slot->get_shape() != like.get_shape() || slot->get_dtype() != like.get_dtype())
    {
        slot = dl::Tensor(like.get_shape(), like.get_device(), like.get_dtype());
        fill_zero(*slot);
    }
    return *slot;
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

auto spatial_out(int in_size, int kernel, int stride, int padding) -> int
{
    return ((in_size + (2 * padding) - kernel) / stride) + 1;
}

__device__ auto weight_index(int loc, int oc, int ic, int kh, int kw, int out_channels, int in_channels, int kernel)
    -> int
{
    return (((((((loc * out_channels) + oc) * in_channels) + ic) * kernel) + kh) * kernel) + kw;
}

__global__ void local_forward_kernel(const float* input, const float* weights, const float* biases, float* output,
    int batch, int in_channels, int in_h, int in_w, int out_channels, int out_h, int out_w, int kernel, int stride,
    int padding)
{
    const int count = batch * out_channels * out_h * out_w;
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count)
    {
        return;
    }

    const int loc_count = out_h * out_w;
    const int ow = index % out_w;
    const int oh = (index / out_w) % out_h;
    const int oc = (index / (out_w * out_h)) % out_channels;
    const int n = index / (out_w * out_h * out_channels);
    const int loc = (oh * out_w) + ow;

    float acc = biases[(oc * loc_count) + loc];
    for (int ic = 0; ic < in_channels; ++ic)
    {
        for (int kh = 0; kh < kernel; ++kh)
        {
            const int ih = (oh * stride) + kh - padding;
            if (ih < 0 || ih >= in_h)
            {
                continue;
            }
            for (int kw = 0; kw < kernel; ++kw)
            {
                const int iw = (ow * stride) + kw - padding;
                if (iw < 0 || iw >= in_w)
                {
                    continue;
                }
                const int in_idx = ((((n * in_channels) + ic) * in_h) + ih) * in_w + iw;
                const int w_idx = weight_index(loc, oc, ic, kh, kw, out_channels, in_channels, kernel);
                acc += input[in_idx] * weights[w_idx];
            }
        }
    }
    output[index] = acc;
}

__global__ void local_backward_kernel(const float* input, const float* weights, const float* grad_output,
    float* grad_input, float* grad_weights, float* grad_biases, int batch, int in_channels, int in_h, int in_w,
    int out_channels, int out_h, int out_w, int kernel, int stride, int padding)
{
    const int count = batch * out_channels * out_h * out_w;
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count)
    {
        return;
    }

    const int loc_count = out_h * out_w;
    const int ow = index % out_w;
    const int oh = (index / out_w) % out_h;
    const int oc = (index / (out_w * out_h)) % out_channels;
    const int n = index / (out_w * out_h * out_channels);
    const int loc = (oh * out_w) + ow;
    const float go = grad_output[index];

    atomicAdd(grad_biases + ((oc * loc_count) + loc), go);

    for (int ic = 0; ic < in_channels; ++ic)
    {
        for (int kh = 0; kh < kernel; ++kh)
        {
            const int ih = (oh * stride) + kh - padding;
            if (ih < 0 || ih >= in_h)
            {
                continue;
            }
            for (int kw = 0; kw < kernel; ++kw)
            {
                const int iw = (ow * stride) + kw - padding;
                if (iw < 0 || iw >= in_w)
                {
                    continue;
                }
                const int in_idx = ((((n * in_channels) + ic) * in_h) + ih) * in_w + iw;
                const int w_idx = weight_index(loc, oc, ic, kh, kw, out_channels, in_channels, kernel);
                atomicAdd(grad_weights + w_idx, go * input[in_idx]);
                atomicAdd(grad_input + in_idx, go * weights[w_idx]);
            }
        }
    }
}

auto launch_count(int count) -> dim3
{
    return dim3(static_cast<unsigned int>((count + kLocalThreads - 1) / kLocalThreads));
}

} // namespace

LocalLayer::LocalLayer(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val,
    int in_height, int in_width)
    : weights_({ spatial_out(in_height, kernel_size, stride_val, padding_val)
                     * spatial_out(in_width, kernel_size, stride_val, padding_val),
                   out_channels, in_channels, kernel_size, kernel_size },
          dl::Device::GPU)
    , biases_({ 1, out_channels, spatial_out(in_height, kernel_size, stride_val, padding_val),
          spatial_out(in_width, kernel_size, stride_val, padding_val) },
          dl::Device::GPU)
    , weights_gradient_({ spatial_out(in_height, kernel_size, stride_val, padding_val)
                              * spatial_out(in_width, kernel_size, stride_val, padding_val),
                            out_channels, in_channels, kernel_size, kernel_size },
          dl::Device::GPU)
    , biases_gradient_({ 1, out_channels, spatial_out(in_height, kernel_size, stride_val, padding_val),
          spatial_out(in_width, kernel_size, stride_val, padding_val) },
          dl::Device::GPU)
    , in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride_val)
    , padding_(padding_val)
    , in_height_(in_height)
    , in_width_(in_width)
    , out_height_(spatial_out(in_height, kernel_size, stride_val, padding_val))
    , out_width_(spatial_out(in_width, kernel_size, stride_val, padding_val))
{
    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 || stride_val <= 0 || padding_val < 0
        || in_height <= 0 || in_width <= 0 || out_height_ <= 0 || out_width_ <= 0)
    {
        throw std::runtime_error("LocalLayer received invalid constructor arguments");
    }

    device_ = dl::Device::GPU;
    const float fan_in = static_cast<float>(in_channels_ * kernel_size_ * kernel_size_);
    const float bound = std::sqrt(2.0F / fan_in);
    fill_uniform(weights_, -bound, bound, 0x10CA1ULL);
    fill_zero(biases_);
    fill_zero(weights_gradient_);
    fill_zero(biases_gradient_);
}

auto LocalLayer::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("LocalLayer_Forward");
    const dl::StreamGuard stream_guard(stream);
    if (input_tensor.get_shape().size() != 4)
    {
        throw std::runtime_error("LocalLayer expects NCHW input, got " + input_tensor.describe());
    }
    if (input_tensor.get_shape()[1] != in_channels_ || input_tensor.get_shape()[2] != in_height_
        || input_tensor.get_shape()[3] != in_width_)
    {
        throw std::runtime_error("LocalLayer input shape mismatch: " + input_tensor.describe());
    }

    const int batch = input_tensor.get_shape()[0];
    const std::vector<int> out_shape { batch, out_channels_, out_height_, out_width_ };
    dl::Tensor& output = dl::Tensor::ensure(output_cache_, out_shape, dl::Device::GPU, input_tensor.get_dtype());
    input_cache_ = input_tensor.view(input_tensor.get_shape());
    input_cache_ready_ = true;

    const int count = batch * out_channels_ * out_height_ * out_width_;
    local_forward_kernel<<<launch_count(count), kLocalThreads, 0, dl::current_stream()>>>(input_tensor.data(),
        weights_.data(), biases_.data(), output.data(), batch, in_channels_, in_height_, in_width_, out_channels_,
        out_height_, out_width_, kernel_size_, stride_, padding_);
    CHECK_CUDA(cudaGetLastError());
    return output.as_view();
}

auto LocalLayer::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("LocalLayer_Backward");
    const dl::StreamGuard stream_guard(stream);
    if (!input_cache_ready_ || !input_cache_.has_value())
    {
        throw std::runtime_error("LocalLayer::backward requires a matching forward");
    }

    const int batch = input_cache_->get_shape()[0];
    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, input_cache_->get_shape(), dl::Device::GPU,
        input_cache_->get_dtype());
    fill_zero(grad_input);
    fill_zero(weights_gradient_);
    fill_zero(biases_gradient_);

    const int count = batch * out_channels_ * out_height_ * out_width_;
    local_backward_kernel<<<launch_count(count), kLocalThreads, 0, dl::current_stream()>>>(input_cache_->data(),
        weights_.data(), output_error_derivative.data(), grad_input.data(), weights_gradient_.data(),
        biases_gradient_.data(), batch, in_channels_, in_height_, in_width_, out_channels_, out_height_, out_width_,
        kernel_size_, stride_, padding_);
    CHECK_CUDA(cudaGetLastError());
    input_cache_ready_ = false;
    return grad_input.as_view();
}

void LocalLayer::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("LocalLayer_Step");
    const dl::StreamGuard stream_guard(stream);
    if (frozen())
    {
        return;
    }
    const float clip = parameter_clip_bound();
    const float lr = scaled_learning_rate();
    if (momentum > 0.0F)
    {
        dl::Tensor& weight_velocity = ensure_zero_like(weights_velocity_, weights_);
        dl::Tensor& bias_velocity = ensure_zero_like(biases_velocity_, biases_);
        weights_.sgd_momentum_update_(weights_gradient_, weight_velocity, lr, momentum, kWeightDecay, clip);
        biases_.sgd_momentum_update_(biases_gradient_, bias_velocity, lr, momentum, kWeightDecay, clip);
        return;
    }
    weights_.sgd_update_(weights_gradient_, lr, kWeightDecay, clip);
    biases_.sgd_update_(biases_gradient_, lr, kWeightDecay, clip);
}

void LocalLayer::clip_gradients(float abs_bound, cudaStream_t stream)
{
    const dl::StreamGuard stream_guard(stream);
    if (abs_bound <= 0.0F)
    {
        return;
    }
    weights_gradient_.clamp_(-abs_bound, abs_bound);
    biases_gradient_.clamp_(-abs_bound, abs_bound);
}

auto LocalLayer::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights", weights_.view(weights_.get_shape()));
    params.emplace("bias", biases_.view(biases_.get_shape()));
    return params;
}

void LocalLayer::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    copy_same_size(weights_, params.at("weights"), "LocalLayer::set_parameters weights");
    copy_same_size(biases_, params.at("bias"), "LocalLayer::set_parameters bias");
}

auto LocalLayer::to(dl::Device device) -> void
{
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("LocalLayer parameters must remain on the GPU");
    }
    device_ = device;
}
