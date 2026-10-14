#pragma once

#include <torch/torch.h>

#include "common/torch_paired_init.hpp"

/**
 * LibTorch twin of TabularMLP: Linear, LeakyReLU(0.1), Linear.
 */
struct TorchTabularMLPImpl : torch::nn::Module
{
    torch::nn::Linear fc1 { nullptr };
    torch::nn::Linear fc2 { nullptr };

    TorchTabularMLPImpl(int in_features, int hidden_size, int num_classes)
    {
        fc1 = register_module("fc1", torch::nn::Linear(in_features, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, num_classes));
        align_torch_linear(fc1);
        align_torch_linear(fc2);
    }

    auto forward(torch::Tensor input) -> torch::Tensor
    {
        return fc2->forward(torch::leaky_relu(fc1->forward(input), 0.1));
    }
};
TORCH_MODULE(TorchTabularMLP);
