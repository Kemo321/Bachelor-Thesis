#pragma once

#include "../cnn_shapes.hpp"
#include "../common/torch_paired_init.hpp"

#include <torch/torch.h>

/**
 * LibTorch twin of FusedCbrCNN.
 *
 * Each block is Conv2d (bias on), BatchNorm2d(eps=1e-5, momentum=0.1), LeakyReLU(0.1), then MaxPool 2.
 * That is the unfused form of FusedCBR2d. The head is the same linear layer.
 */
struct TorchFusedCbrCNNImpl : torch::nn::Module
{
    torch::nn::Conv2d conv1 { nullptr };
    torch::nn::BatchNorm2d bn1 { nullptr };
    torch::nn::Conv2d conv2 { nullptr };
    torch::nn::BatchNorm2d bn2 { nullptr };
    torch::nn::Linear fc { nullptr };

    TorchFusedCbrCNNImpl(int num_classes, int image_size, int in_channels)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 16, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(16).eps(1.0e-5).momentum(0.1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32).eps(1.0e-5).momentum(0.1)));
        fc = register_module("fc", torch::nn::Linear(pooled_cnn_features(image_size), num_classes));
        align_torch_conv(conv1);
        align_torch_batchnorm(bn1);
        align_torch_conv(conv2);
        align_torch_batchnorm(bn2);
        align_torch_linear(fc);
    }

    auto forward(torch::Tensor input) -> torch::Tensor
    {
        auto hidden = torch::leaky_relu(bn1->forward(conv1->forward(input)), 0.1);
        hidden = torch::max_pool2d(hidden, 2);
        hidden = torch::leaky_relu(bn2->forward(conv2->forward(hidden)), 0.1);
        hidden = torch::max_pool2d(hidden, 2);
        return fc->forward(hidden.flatten(1));
    }
};
TORCH_MODULE(TorchFusedCbrCNN);
