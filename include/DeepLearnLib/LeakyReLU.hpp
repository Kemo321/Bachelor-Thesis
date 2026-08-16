#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <optional>

/**
 * LeakyReLU activation implemented with Thrust on the GPU.
 */
class LeakyReLU : public Layer
{
public:
    explicit LeakyReLU(float slope_val = 0.1F);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;

private:
    std::optional<dl::Tensor> input_cache_;
    float slope_;
};
