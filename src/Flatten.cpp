#include "DeepLearnLib/Flatten.hpp"

/**
 * @brief Computes the forward pass for the Flatten layer.
 *
 * @param input_tensor The input tensor to the layer. Expected shape: [Batch, ...].
 * @return A flattened tensor with shape [Batch, FlattenedSize].
 */

auto Flatten::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    input_shape_cache_ = input_tensor.sizes().vec();
    int64_t batch_size = input_tensor.size(0);
    int64_t flattened_size = 1;
    for (size_t i = 1; i < input_shape_cache_.size(); ++i)
    {
        flattened_size *= input_shape_cache_[i];
    }
    return input_tensor.view({batch_size, flattened_size}).contiguous();
}

/**
 * @brief Computes the backward pass for the Flatten layer.
 *
 * @param output_error_derivative The derivative of the error with respect to the output. Expected shape: [Batch, FlattenedSize].
 * @return The derivative of the error with respect to the input. Expected shape: [Batch, ...].
 * @note Reshapes the gradient back to the original input shape cache.
 */

auto Flatten::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    return output_error_derivative.view(input_shape_cache_);
}