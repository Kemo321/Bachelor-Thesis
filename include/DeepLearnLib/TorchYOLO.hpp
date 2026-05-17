#pragma once

#include <torch/torch.h>

struct YOLOv1Impl : torch::nn::Module
{
    torch::nn::Sequential backbone{nullptr};
    torch::nn::Sequential head{nullptr};

    YOLOv1Impl(int num_classes = 20);

    [[nodiscard]] auto forward(torch::Tensor input_tensor) -> torch::Tensor;

private:
    int num_classes_;
};

TORCH_MODULE(YOLOv1);

[[nodiscard]] auto compute_yolo_loss(const torch::Tensor& prediction, const torch::Tensor& target) -> torch::Tensor;