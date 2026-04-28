#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>


class Dropout : public Layer
{
public:
    explicit Dropout(float probability = 0.5F);

    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;
    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

private:
    float probability_;
    torch::Tensor mask_;
};