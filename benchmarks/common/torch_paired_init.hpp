#pragma once

#include "torch_optim.hpp"

#include "DeepLearnLib/layers/Gru.hpp"
#include "DeepLearnLib/utils/ExperimentSeed.hpp"

#include <torch/torch.h>

#include <stdexcept>
#include <string>
#include <vector>

/**
 * Copies the custom layer's deterministic initialisation into the LibTorch twin.
 * Conv and linear layers share `paired_uniform`. The GRU copies the cuDNN weight
 * space, because that layout is not a plain torch::nn::GRU parameter tensor.
 */

inline auto torch_parameter(torch::nn::Module& module, const std::string& name) -> torch::Tensor
{
    for (auto& item : module.named_parameters())
    {
        if (item.key() == name)
        {
            return item.value();
        }
    }
    throw std::runtime_error("Torch module has no parameter named " + name);
}

inline auto copy_uniform_tensor(torch::Tensor parameter, float bound, unsigned long long seed) -> void
{
    const auto contiguous = parameter.contiguous();
    std::vector<float> host(static_cast<std::size_t>(contiguous.numel()));
    for (int index = 0; index < static_cast<int>(host.size()); ++index)
    {
        host[static_cast<std::size_t>(index)] = dl::paired_uniform(index, -bound, bound, seed);
    }
    const torch::Tensor values = torch::from_blob(host.data(), contiguous.sizes(), torch::kFloat32).clone();
    parameter.copy_(values.view(parameter.sizes()));
}

inline auto align_torch_conv(torch::nn::Conv2d& conv) -> void
{
    prepare_torch_experiment();
    torch::NoGradGuard guard;
    const auto weight = conv->weight;
    const float fan_in = static_cast<float>(weight.size(1) * weight.size(2) * weight.size(3));
    const float bound = dl::uniform_bound(fan_in);
    copy_uniform_tensor(weight, bound, dl::kConvWeightSeed);
    copy_uniform_tensor(conv->bias, bound, dl::kConvBiasSeed);
}

inline auto align_torch_linear(torch::nn::Linear& linear) -> void
{
    prepare_torch_experiment();
    torch::NoGradGuard guard;
    const int out_features = static_cast<int>(linear->weight.size(0));
    const int in_features = static_cast<int>(linear->weight.size(1));
    const float bound = dl::uniform_bound(static_cast<float>(in_features));
    std::vector<float> host(static_cast<std::size_t>(in_features * out_features));
    for (int index = 0; index < static_cast<int>(host.size()); ++index)
    {
        host[static_cast<std::size_t>(index)] = dl::paired_uniform(index, -bound, bound, dl::kLinearWeightSeed);
    }
    // Custom FullyConnected stores [in, out]. Torch stores [out, in].
    const torch::Tensor custom_layout
        = torch::from_blob(host.data(), { in_features, out_features }, torch::kFloat32).clone();
    linear->weight.copy_(custom_layout.transpose(0, 1).contiguous());

    std::vector<float> bias(static_cast<std::size_t>(out_features));
    for (int index = 0; index < out_features; ++index)
    {
        bias[static_cast<std::size_t>(index)] = dl::paired_uniform(index, -bound, bound, dl::kLinearBiasSeed);
    }
    const torch::Tensor bias_values = torch::from_blob(bias.data(), { out_features }, torch::kFloat32).clone();
    linear->bias.copy_(bias_values);
}

inline auto align_torch_batchnorm(torch::nn::BatchNorm2d& norm) -> void
{
    torch::NoGradGuard guard;
    norm->weight.fill_(1.0);
    norm->bias.zero_();
    norm->running_mean.zero_();
    norm->running_var.fill_(1.0);
    if (norm->num_batches_tracked.defined())
    {
        norm->num_batches_tracked.zero_();
    }
}

inline auto copy_rows(torch::Tensor destination, int row, const std::vector<float>& values, int rows, int cols) -> void
{
    const torch::Tensor host
        = torch::from_blob(const_cast<float*>(values.data()), { rows, cols }, torch::kFloat32).clone();
    destination.slice(0, row, row + rows).copy_(host);
}

inline auto align_torch_gru(torch::nn::GRU& gru, int input_size, int hidden_size) -> void
{
    prepare_torch_experiment();
    Gru source(input_size, hidden_size);
    const std::vector<Gru::LinearLayer> gates = source.export_linear_layers();
    if (gates.size() != 6U)
    {
        throw std::runtime_error("cuDNN GRU did not expose six linear layers");
    }

    torch::NoGradGuard guard;
    torch::Tensor weight_ih = torch_parameter(*gru, "weight_ih_l0");
    torch::Tensor weight_hh = torch_parameter(*gru, "weight_hh_l0");
    torch::Tensor bias_ih = torch_parameter(*gru, "bias_ih_l0");
    torch::Tensor bias_hh = torch_parameter(*gru, "bias_hh_l0");

    auto install = [&](int source_gate, int row_slot, torch::Tensor& weight, torch::Tensor& bias, int cols)
    {
        const Gru::LinearLayer& layer = gates[static_cast<std::size_t>(source_gate)];
        if (layer.rows != hidden_size || layer.cols != cols)
        {
            throw std::runtime_error("cuDNN GRU gate shape does not match torch::nn::GRU");
        }
        copy_rows(weight, row_slot * hidden_size, layer.matrix, layer.rows, layer.cols);
        if (static_cast<int>(layer.bias.size()) == hidden_size)
        {
            const torch::Tensor host
                = torch::from_blob(const_cast<float*>(layer.bias.data()), { hidden_size }, torch::kFloat32).clone();
            bias.slice(0, row_slot * hidden_size, (row_slot + 1) * hidden_size).copy_(host);
        }
        else if (layer.bias.empty())
        {
            bias.slice(0, row_slot * hidden_size, (row_slot + 1) * hidden_size).zero_();
        }
        else
        {
            throw std::runtime_error("cuDNN GRU bias length does not match the hidden size");
        }
    };

    // cuDNN linLayerID 0..2 are reset, update, candidate. That is PyTorch's row order.
    for (int gate = 0; gate < 3; ++gate)
    {
        install(gate, gate, weight_ih, bias_ih, input_size);
    }
    for (int gate = 0; gate < 3; ++gate)
    {
        install(gate + 3, gate, weight_hh, bias_hh, hidden_size);
    }
    gru->flatten_parameters();
}
