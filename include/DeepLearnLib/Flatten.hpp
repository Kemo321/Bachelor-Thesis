#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>
#include <vector>
#include <cstdint>

/**
 * @brief Flatten layer that collapses all dimensions except the batch dimension.
 *
 * This layer is typically used to transition from convolutional feature maps
 * to fully-connected layers. The forward pass reshapes an input tensor with
 * shape [Batch, C, H, W] (or generally [Batch, ...]) to [Batch, N] where
 * N = product of non-batch dimensions. The backward pass reshapes the
 * gradient from [Batch, N] back to the cached input shape.
 */
class Flatten : public Layer
{
public:
    Flatten() = default;

    /**
     * @brief Forward pass that flattens the input tensor.
     *
     * @param inputTensor Input tensor with shape [Batch, ...] (e.g. [B, C, H, W]).
     *                    The batch dimension (first dim) is preserved.
     * @return torch::Tensor Output tensor with shape [Batch, N] where
     *         N = product of input dimensions except the batch dimension.
     *
     * @note Implementations should ensure the returned tensor is contiguous
     *       (e.g., call .contiguous()) after any view/permute operations to
     *       avoid potential memory-stride issues on CUDA.
     */
    [[nodiscard]] auto forward(const torch::Tensor& inputTensor) -> torch::Tensor override;

    /**
     * @brief Backward pass that restores the gradient to the input shape.
     *
     * @param outputErrorDerivative Gradient from subsequent layer with shape
     *                              [Batch, N] (matching forward output).
     * @return torch::Tensor Gradient reshaped to the original input shape
     *                      cached during forward (e.g. [B, C, H, W]).
     *
     * @note The method should call .contiguous() after any reshape/view
     *       operations if required by downstream code.
     */
    [[nodiscard]] auto backward(const torch::Tensor& outputErrorDerivative) -> torch::Tensor override;

private:
    /// Cached input shape recorded during forward: [Batch, ...]
    std::vector<int64_t> input_shape_cache_;
};