#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <optional>

/**
 * Inverted dropout on the GPU.
 *
 * Training samples a Bernoulli mask (Thrust hash PRNG) and scales by 1 / (1 - p).
 * Evaluation is identity.
 */
class Dropout : public Layer
{
public:
    explicit Dropout(float probability = 0.5F);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;

private:
    float probability_;
    unsigned long long seed_;
    std::optional<dl::Tensor> mask_;
    std::optional<dl::Tensor> output_cache_;
    std::optional<dl::Tensor> grad_input_cache_;
    bool mask_ready_ { false };
};
