#pragma once

#include <torch/torch.h>

struct YOLOv1Impl : torch::nn::Module
{
    torch::nn::Sequential backbone{nullptr};
    torch::nn::Sequential head{nullptr};

    YOLOv1Impl();

    [[nodiscard]] auto forward(torch::Tensor input_tensor) -> torch::Tensor;
};

TORCH_MODULE(YOLOv1);

[[nodiscard]] auto compute_yolo_loss(const torch::Tensor& prediction, const torch::Tensor& target) -> torch::Tensor;