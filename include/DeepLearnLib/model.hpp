#pragma once

#include <torch/torch.h>

struct YOLOv1Impl : torch::nn::Module
{
    torch::nn::Sequential backbone{ nullptr };
    torch::nn::Sequential head{ nullptr };

    YOLOv1Impl();
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(YOLOv1);

torch::Tensor compute_yolo_loss(const torch::Tensor& pred, const torch::Tensor& target);