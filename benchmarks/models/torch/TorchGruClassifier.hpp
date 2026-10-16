#pragma once

#include <torch/torch.h>
#include <tuple>

#include "common/torch_paired_init.hpp"

struct TorchGruClassifierImpl : torch::nn::Module
{
    torch::nn::GRU gru { nullptr };
    torch::nn::Linear fc { nullptr };

    TorchGruClassifierImpl(int input_size, int hidden_size, int num_classes)
    {
        gru = register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(input_size, hidden_size).batch_first(true)));
        fc = register_module("fc", torch::nn::Linear(hidden_size, num_classes));
        align_torch_gru(gru, input_size, hidden_size);
        align_torch_linear(fc);
    }

    auto forward(torch::Tensor input) -> torch::Tensor
    {
        gru->flatten_parameters();
        torch::Tensor sequence;
        torch::Tensor hidden;
        std::tie(sequence, hidden) = gru->forward(input);
        return fc->forward(hidden.squeeze(0));
    }
};
TORCH_MODULE(TorchGruClassifier);
