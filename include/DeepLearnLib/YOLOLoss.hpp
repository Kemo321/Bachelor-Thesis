#pragma once
#include <torch/torch.h>

class YOLOLoss {
public:
    [[nodiscard]] static auto loss(const torch::Tensor& target, const torch::Tensor& prediction, int num_classes = 20) -> torch::Tensor;
    [[nodiscard]] static auto loss_derivative(const torch::Tensor& target, const torch::Tensor& prediction, int num_classes = 20) -> torch::Tensor;
private:
    static auto calculate_iou(const torch::Tensor& box1, const torch::Tensor& box2) -> torch::Tensor;
};