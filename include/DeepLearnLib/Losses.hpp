#pragma once

#include "DeepLearnLib/Tensor.hpp"

/**
 * Mean squared error over all elements: mean((prediction - target)^2).
 */
class MSELoss
{
public:
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
};

/**
 * Mean softmax-cross-entropy over a rank-2 batch of logits [N, C].
 *
 * Targets are dense one-hot (or probability) tensors of the same shape.
 * Softmax is fused into both the forward loss and the gradient.
 */
class CrossEntropyLoss
{
public:
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
};
