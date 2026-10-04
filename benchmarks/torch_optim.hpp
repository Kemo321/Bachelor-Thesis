#pragma once

#include <cstring>
#include <torch/torch.h>
#include <vector>

inline auto make_sgd(torch::nn::Module& model, double learning_rate, double momentum, double weight_decay)
    -> torch::optim::SGD
{
    return torch::optim::SGD(
        model.parameters(), torch::optim::SGDOptions(learning_rate).momentum(momentum).weight_decay(weight_decay));
}

inline auto set_sgd_lr(torch::optim::Optimizer& optimizer, double learning_rate) -> void
{
    for (auto& group : optimizer.param_groups())
    {
        static_cast<torch::optim::SGDOptions&>(group.options()).lr(learning_rate);
    }
}

/** Element-wise |g| clip; `clip <= 0` is a no-op (matches custom Network). */
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
