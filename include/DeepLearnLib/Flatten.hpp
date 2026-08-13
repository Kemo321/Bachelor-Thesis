#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <vector>

/**
 * @brief Flatten layer that collapses all dimensions except the batch dimension.
 *
 * Forward reshapes [Batch, ...] to [Batch, N]. Backward restores the cached input shape.
 */
class Flatten : public Layer
{
public:
    Flatten() = default;

    /**
     * @brief Forward pass that flattens the input tensor.
     *
     * @param input_tensor Input tensor with shape [Batch, ...].
     * @return Output tensor with shape [Batch, N].
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;

    /**
     * @brief Backward pass that restores the gradient to the input shape.
     *
     * @param output_error_derivative Gradient with shape [Batch, N].
     * @return Gradient reshaped to the cached input shape.
     */
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;

private:
    std::vector<int> input_shape_cache_;
};
