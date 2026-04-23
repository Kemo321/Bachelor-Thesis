#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>

class Conv2d : public Layer
{
public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val, float inertia_val = 0.0F);

    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;
    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

private:
    torch::Tensor weights_;
    torch::Tensor biases_;
    torch::Tensor input_cache_;
    torch::Tensor weights_gradient_;
    int stride_;
    int padding_;
    float inertia_;
};
