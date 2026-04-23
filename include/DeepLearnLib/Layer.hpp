#pragma once
#include <torch/torch.h>

class Layer
{
public:
    float learning_rate = 0.001F;

    virtual ~Layer() = default;

    [[nodiscard]] virtual torch::Tensor forward(const torch::Tensor& x) = 0;
    [[nodiscard]] virtual torch::Tensor backward(const torch::Tensor& output_error_derivative) = 0;
};
