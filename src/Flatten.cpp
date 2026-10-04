#include "DeepLearnLib/Flatten.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <stdexcept>
#include <string>

namespace
{

// Rejects a host tensor or a null device pointer. The flatten is only a view
// and does not copy data on the GPU.
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

// Changes the view to [batch, rest] without copying data. The input shape stays
// in the cache because backward has to restore it.
auto Flatten::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Flatten_Forward");
    (void)stream;
    require_gpu(input_tensor, "Flatten::forward input");
    if (input_tensor.get_shape().empty())
    {
        throw std::runtime_error("Flatten::forward requires a tensor with a batch dimension");
    }

    // Saved so backward can restore this rank. The returned tensor is a view, not a copy.
    input_shape_cache_ = input_tensor.get_shape();
    const int batch_size = input_shape_cache_.front();
    if (batch_size <= 0)
    {
        throw std::runtime_error("Flatten::forward requires a positive batch size");
    }

    const int flattened_size = static_cast<int>(input_tensor.get_size() / static_cast<size_t>(batch_size));
    return input_tensor.view({ batch_size, flattened_size });
}

// Restores the gradient to the shape from before the flatten. This is only a view:
// the layer has no weights and does not change the values.
auto Flatten::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Flatten_Backward");
    (void)stream;
    require_gpu(output_error_derivative, "Flatten::backward grad_output");
    if (input_shape_cache_.empty())
    {
        throw std::runtime_error("Flatten::backward requires a preceding forward pass");
    }
    // View only. The values are the incoming gradient, reshaped to the cached input.
    return output_error_derivative.view(input_shape_cache_);
}
