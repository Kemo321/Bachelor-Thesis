#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>

/**
 * @class LeakyReLU
 * @brief Implements the LeakyReLU activation function as a neural network layer.
 * 
 * This layer applies the LeakyReLU activation function, which allows a small, non-zero gradient when the unit is not active.
 */
class LeakyReLU : public Layer
{
public:
    /**
     * @brief Constructs a LeakyReLU layer with a specified slope value.
     * @param slopeVal The slope value for negative inputs. Default is 0.1F.
     */
    explicit LeakyReLU(float slopeVal = 0.1F);

    /**
     * @brief Performs the forward pass of the LeakyReLU activation function.
     * @param inputTensor The input tensor. Expected shape: [Batch, Channels, Height, Width].
     * @return The output tensor after applying the LeakyReLU activation. Shape: [Batch, Channels, Height, Width].
     */
    [[nodiscard]] auto forward(const torch::Tensor& inputTensor) -> torch::Tensor override;

    /**
     * @brief Performs the backward pass, computing the gradient of the loss with respect to the input.
     * @param outputErrorDerivative The gradient of the loss with respect to the output. Shape: [Batch, Channels, Height, Width].
     * @return The gradient of the loss with respect to the input. Shape: [Batch, Channels, Height, Width].
     * @note This method applies the chain rule to compute the gradient.
     */
    [[nodiscard]] auto backward(const torch::Tensor& outputErrorDerivative) -> torch::Tensor override;

private:
    torch::Tensor input_cache_; ///< Cached input tensor for use in the backward pass.
    float slope_; ///< Slope value for the LeakyReLU activation function.
};
