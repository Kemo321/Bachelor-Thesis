#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>

/**
 * @brief Dropout regularization layer.
 *
 * @details During training this layer randomly zeroes some of the elements of the
 * input tensor with probability `probability_`. To preserve the expected value
 * of activations the remaining elements are scaled by 1 / (1 - probability_).
 * The layer stores a binary mask (same shape as the input) that is reused in
 * the backward pass to propagate gradients only through the retained elements.
 */
class Dropout : public Layer
{
public:
    /**
     * @brief Construct a Dropout layer.
     *
     * @param probability Probability of an element to be zeroed during training.
     *                    Value in range [0.0F, 1.0F].
     */
    explicit Dropout(float probability = 0.5F);

    /**
     * @brief Forward pass of the Dropout layer.
     *
     * @param input_tensor Input tensor of arbitrary shape. Typical shape examples:
     *                     - Fully-connected: [Batch, Features]
     *                     - Convolutional: [Batch, Channels, Height, Width]
     * @return Tensor with the same shape as input_tensor where a subset of
     *         elements have been zeroed and the remainder scaled by
     *         1 / (1 - probability_). Shape: same as input_tensor.
     * @note In training mode a binary mask is sampled from Bernoulli(1 - probability_)
     *       and applied. In evaluation/inference mode the input is returned
     *       unchanged (but implementations should ensure torch::NoGradGuard is used
     *       externally when needed).
     */
    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;

    /**
     * @brief Backward pass of the Dropout layer.
     *
     * @param output_error_derivative Gradient tensor with respect to the layer's
     *                                output. Shape: same as the forward input.
     * @return Gradient tensor with respect to the layer's input. Shape: same as input.
     * @note The backward pass applies the same binary mask as in the forward pass
     *       and scales gradients accordingly. This implements the chain rule for
     *       masked activations.
     */
    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

private:
    /**
     * @brief Probability of dropping an element during training.
     * @note Value is in the range [0.0F, 1.0F].
     */
    float probability_;

    /**
     * @brief Binary mask sampled during the forward pass.
     *
     * The mask has the same shape as the input tensor passed to forward and is
     * reused in backward to mask gradients. Stored as a floating-point tensor
     * with values 0.0F or 1.0F for numerical compatibility.
     */
    torch::Tensor mask_;
};