#pragma once

#include "../cnn_shapes.hpp"
#include "../common/torch_paired_init.hpp"

#include <torch/torch.h>

/**
 * LibTorch twin of SimpleCNN.
 *
 * Conv  C→16 3×3 pad 1, LeakyReLU 0.1, MaxPool 2,
 * Conv 16→32 3×3 pad 1, LeakyReLU 0.1, MaxPool 2, Linear onto the classes.
 */
struct TorchSimpleCNNImpl : torch::nn::Module
{
    torch::nn::Conv2d conv1 { nullptr };
    torch::nn::Conv2d conv2 { nullptr };
    torch::nn::Linear fc { nullptr };

    TorchSimpleCNNImpl(int num_classes, int image_size, int in_channels)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 16, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)));
        const int features = pooled_cnn_features(image_size);
        fc = register_module("fc", torch::nn::Linear(features, num_classes));
        align_torch_conv(conv1);
        align_torch_conv(conv2);
        align_torch_linear(fc);
    }

    auto forward(torch::Tensor input) -> torch::Tensor
    {
        auto hidden = torch::leaky_relu(conv1->forward(input), 0.1);
        hidden = torch::max_pool2d(hidden, 2);
        hidden = torch::leaky_relu(conv2->forward(hidden), 0.1);
        hidden = torch::max_pool2d(hidden, 2);
        return fc->forward(hidden.flatten(1));
    }
};
TORCH_MODULE(TorchSimpleCNN);
