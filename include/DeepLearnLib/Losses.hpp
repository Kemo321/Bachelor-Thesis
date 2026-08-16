#pragma once

#include "DeepLearnLib/Tensor.hpp"

/**
 * @brief Mean squared error over all elements: mean((prediction - target)^2).
 */
class MSELoss
{
public:
    /**
     * @brief Scalar GPU tensor of shape [1].
     */
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;

    /**
     * @brief dL/d(prediction) with mean reduction: 2 * (prediction - target) / N.
     */
    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
};

/**
 * @brief Mean softmax-cross-entropy over a rank-2 batch of logits [N, C].
 *
 * Targets are dense one-hot (or probability) tensors of the same shape.
 * Softmax is fused into both the forward loss and the gradient.
 */
class CrossEntropyLoss
{
public:
    /**
     * @brief Scalar GPU tensor of shape [1]: mean_n sum_c -target * log(softmax(pred)).
     */
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;

    /**
     * @brief dL/d(logits) = (softmax(pred) - target) / N.
     */
    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
};
