#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <optional>

/**
 * @brief Dropout regularization layer with inverted dropout on the GPU.
 *
 * During training, a Bernoulli mask is sampled with a hash-based PRNG via Thrust
 * and applied with scale 1 / (1 - probability). Evaluation leaves the input unchanged.
 */
class Dropout : public Layer
{
public:
    /**
     * @brief Construct a Dropout layer.
     * @param probability Probability of dropping an element in [0, 1).
     */
    explicit Dropout(float probability = 0.5F);

    /**
     * @brief Forward pass of Dropout.
     * @param input_tensor Input of arbitrary shape.
     * @return Masked and scaled tensor in training, or a view of the input in eval.
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;

    /**
     * @brief Backward pass of Dropout.
     * @param output_error_derivative Upstream gradient, same shape as the forward input.
     * @return Gradient masked with the cached scaled Bernoulli mask.
     */
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;

private:
    float probability_;
    unsigned long long seed_;
    std::optional<dl::Tensor> mask_;
};
