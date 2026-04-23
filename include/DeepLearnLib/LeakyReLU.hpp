#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>

class LeakyReLU : public Layer
{
public:
    explicit LeakyReLU(float slope_val = 0.1F);

    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;

    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

private:
    torch::Tensor input_cache_;
    float slope_;
};
