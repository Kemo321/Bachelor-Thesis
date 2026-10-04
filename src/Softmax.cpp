#include "DeepLearnLib/Softmax.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <stdexcept>
#include <string>

namespace
{

// Rejects a host tensor or a null device pointer. cuDNN softmax reads the device
// memory named by the descriptor, with no host-to-device copy.
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

// Sets the cuDNN NCHW descriptor for rank 2 or rank 4. Rank 2 is treated as
// [N, C, 1, 1], because channel softmax requires four dimensions.
auto Softmax::configure_descriptor(const dl::Tensor& tensor) -> void
{
    const auto& shape = tensor.get_shape();
    if (shape.size() == 2)
    {
        // cuDNN channel mode wants NCHW, so [N, C] gets H = W = 1.
        tensor_desc_.set_nchw(shape[0], shape[1], 1, 1, cudnn_data_type(tensor.get_dtype()));
    }
    else if (shape.size() == 4)
    {
        tensor_desc_.set_nchw(shape[0], shape[1], shape[2], shape[3], cudnn_data_type(tensor.get_dtype()));
    }
    else
    {
        throw std::runtime_error("Softmax requires rank-2 [N, C] or rank-4 NCHW tensors");
    }

    input_shape_cache_ = shape;
    descriptor_configured_ = true;
}

// Computes accurate softmax in channel mode and keeps the output in the cache.
// cuDNN backward needs the softmax values, not the input alone, and ensure does not allocate when the shape is unchanged.
auto Softmax::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Softmax_Forward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    require_gpu(input_tensor, "Softmax::forward input");
    configure_descriptor(input_tensor);

    dl::Tensor& output = dl::Tensor::ensure(output_cache_, input_tensor.get_shape(), dl::Device::GPU,
        input_tensor.get_dtype());
    if (input_tensor.get_size() == 0)
    {
        output_cache_ready_ = true;
        return output.as_view();
    }

    const float alpha = 1.0F;
    const float beta = 0.0F;
    // Accurate channel softmax. The output is cached because backward differentiates y, not the logits.
    CHECK_CUDNN(cudnnSoftmaxForward(dl::get_cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
        tensor_desc_.get(), input_tensor.data(), &beta, tensor_desc_.get(), output.data()));
    output_cache_ready_ = true;
    return output.as_view();
}

// Computes the softmax derivative from the cached output and the gradient. The
// cache flag is cleared afterwards so a stale y is not differentiated.
auto Softmax::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Softmax_Backward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    if (!output_cache_ready_ || !output_cache_.has_value())
    {
        throw std::runtime_error("Softmax::backward requires a preceding forward pass");
    }
    require_gpu(output_error_derivative, "Softmax::backward grad_output");
    if (output_error_derivative.get_shape() != output_cache_->get_shape())
    {
        throw std::runtime_error("Softmax::backward grad_output shape does not match the cached softmax output");
    }

    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, output_cache_->get_shape(), dl::Device::GPU,
        output_cache_->get_dtype());
    if (output_error_derivative.get_size() == 0)
    {
        output_cache_ready_ = false;
        return grad_input.as_view();
    }

    configure_descriptor(*output_cache_);
    const float alpha = 1.0F;
    const float beta = 0.0F;
    // y is the cached softmax output. cuDNN backward does not recompute it from the input.
    CHECK_CUDNN(cudnnSoftmaxBackward(dl::get_cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
        tensor_desc_.get(), output_cache_->data(), tensor_desc_.get(), output_error_derivative.data(), &beta,
        tensor_desc_.get(), grad_input.data()));
    // Drop the cache so a later backward cannot differentiate a stale y.
    output_cache_ready_ = false;
    return grad_input.as_view();
}
