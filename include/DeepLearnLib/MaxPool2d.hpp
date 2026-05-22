#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>
#include <vector>

/**
 * @brief Implements a 2D Max Pooling layer for neural networks.
 * 
 * This layer performs down-sampling by applying a max pooling operation over input tensors.
 */
class MaxPool2d : public Layer
{
public:
    /**
     * @brief Constructs a MaxPool2d layer.
     * 
     * @param kernel_size_val The size of the pooling kernel (e.g., 2 for a 2x2 kernel).
     * @param stride_val The stride of the pooling operation (e.g., 2 for non-overlapping pooling).
     */
    MaxPool2d(int kernel_size_val, int stride_val);

    /**
     * @brief Performs the forward pass of the max pooling layer.
     * 
     * @param input_tensor The input tensor with shape [Batch, Channels, Height, Width].
     * @return A tensor with shape [Batch, Channels, PooledHeight, PooledWidth] after max pooling.
     */
    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;

    /**
     * @brief Performs the backward pass of the max pooling layer.
     * 
     * @param output_error_derivative The gradient of the loss with respect to the output, 
     *                                with shape [Batch, Channels, PooledHeight, PooledWidth].
     * @return A tensor with shape [Batch, Channels, Height, Width] representing the gradient 
     *         of the loss with respect to the input.
     * 
     * @note This method uses the cached indices from the forward pass to propagate the gradient.
     */
    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

private:
    int kernel_size_; ///< The size of the pooling kernel.
    int stride_; ///< The stride of the pooling operation.
    torch::Tensor indices_cache_; ///< Cached indices from the forward pass for use in backpropagation.
    std::vector<int64_t> input_shape_cache_; ///< Cached input shape from the forward pass.
};