#pragma once
#include "Layer.hpp"
#include <torch/torch.h>

class FullyConnected : public Layer
{
public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    FullyConnected(int input_size, int output_size, float inertia_val = 0.0F);

    [[nodiscard]] torch::Tensor forward(const torch::Tensor& input_tensor) override;
    [[nodiscard]] torch::Tensor backward(const torch::Tensor& output_error_derivative) override;

private:
    torch::Tensor weights_;
    torch::Tensor biases_;
    torch::Tensor input_cache_;
    torch::Tensor weights_gradient_;
    float inertia_;
};
