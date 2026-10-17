#pragma once

#include "DeepLearnLib/utils/ExperimentSeed.hpp"

#include <cstring>
#include <torch/torch.h>
#include <vector>

inline auto prepare_torch_experiment() -> void
{
    torch::manual_seed(dl::kExperimentSeed);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setAllowTF32CuDNN(true);
    at::globalContext().setAllowTF32CuBLAS(true);
}

inline auto make_sgd(torch::nn::Module& model, double learning_rate, double momentum, double weight_decay)
    -> torch::optim::SGD
{
    return torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(learning_rate).momentum(momentum).dampening(0.0).weight_decay(weight_decay).nesterov(false));
}

inline auto set_sgd_lr(torch::optim::Optimizer& optimizer, double learning_rate) -> void
{
    for (auto& group : optimizer.param_groups())
    {
        static_cast<torch::optim::SGDOptions&>(group.options()).lr(learning_rate);
    }
}

inline auto torch_device() -> torch::Device
{
    return torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
}

inline auto sync_torch_device() -> void
{
    if (torch::cuda::is_available())
    {
        torch::cuda::synchronize();
    }
}

inline auto clip_torch_grad_value(torch::nn::Module& model, float clip) -> void
{
    if (clip <= 0.0F)
    {
        return;
    }
    for (auto& parameter : model.parameters())
    {
        if (parameter.grad().defined())
        {
            parameter.grad().clamp_(-clip, clip);
        }
    }
}

inline auto tensor_to_host_f32(const torch::Tensor& tensor) -> std::vector<float>
{
    const auto cpu = tensor.contiguous().to(torch::kCPU).to(torch::kFloat32);
    std::vector<float> host(static_cast<std::size_t>(cpu.numel()));
    std::memcpy(host.data(), cpu.data_ptr<float>(), host.size() * sizeof(float));
    return host;
}
