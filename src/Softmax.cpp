#include "DeepLearnLib/Softmax.hpp"

#include <stdexcept>
#include <string>

namespace
{

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

auto Softmax::configure_descriptor(const dl::Tensor& tensor) -> void
{
    const auto& shape = tensor.get_shape();
    if (descriptor_configured_ && shape == input_shape_cache_)
    {
        return;
    }

    if (shape.size() == 2)
    {
        tensor_desc_.set_nchw(shape[0], shape[1], 1, 1);
    }
    else if (shape.size() == 4)
    {
        tensor_desc_.set_nchw(shape[0], shape[1], shape[2], shape[3]);
    }
    else
    {
        throw std::runtime_error("Softmax requires rank-2 [N, C] or rank-4 NCHW tensors");
    }

    input_shape_cache_ = shape;
    descriptor_configured_ = true;
}

auto Softmax::forward(const dl::Tensor& input_tensor) -> dl::Tensor
{
    require_gpu(input_tensor, "Softmax::forward input");
    configure_descriptor(input_tensor);

    dl::Tensor output(input_tensor.get_shape(), dl::Device::GPU);
    if (input_tensor.get_size() == 0)
    {
        output_cache_ = output.view(output.get_shape());
        return output;
    }

    const float alpha = 1.0F;
    const float beta = 0.0F;
    CHECK_CUDNN(cudnnSoftmaxForward(dl::get_cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
        tensor_desc_.get(), input_tensor.data(), &beta, tensor_desc_.get(), output.data()));
    output_cache_ = dl::Tensor(output.get_shape(), dl::Device::GPU);
    CHECK_CUDA(cudaMemcpy(output_cache_->data(), output.data(), output.get_size() * sizeof(float),
        cudaMemcpyDeviceToDevice));
    return output;
}

auto Softmax::backward(const dl::Tensor& output_error_derivative) -> dl::Tensor
{
    if (!output_cache_.has_value())
    {
        throw std::runtime_error("Softmax::backward requires a preceding forward pass");
    }
    require_gpu(output_error_derivative, "Softmax::backward grad_output");
    if (output_error_derivative.get_shape() != output_cache_->get_shape())
    {
        throw std::runtime_error("Softmax::backward grad_output shape does not match the cached softmax output");
    }

    dl::Tensor grad_input(output_cache_->get_shape(), dl::Device::GPU);
    if (output_error_derivative.get_size() == 0)
    {
        output_cache_.reset();
        return grad_input;
    }

    configure_descriptor(*output_cache_);
    const float alpha = 1.0F;
    const float beta = 0.0F;
    CHECK_CUDNN(cudnnSoftmaxBackward(dl::get_cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
        tensor_desc_.get(), output_cache_->data(), tensor_desc_.get(), output_error_derivative.data(), &beta,
        tensor_desc_.get(), grad_input.data()));
    output_cache_.reset();
    return grad_input;
}
