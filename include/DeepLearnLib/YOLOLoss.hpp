#pragma once
#include <torch/torch.h>

class YOLOLoss
{
public:
    [[nodiscard]] static float loss(const torch::Tensor& y_true, const torch::Tensor& y_pred);
    [[nodiscard]] static torch::Tensor loss_derivative(const torch::Tensor& target, const torch::Tensor& pred);
};
