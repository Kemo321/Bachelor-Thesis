#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <map>
#include <string>
#include <torch/torch.h>

class FullyConnected : public Layer
{
public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    FullyConnected(int input_size, int output_size, float inertia_val = 0.0F);

    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;
    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

    auto get_parameters() -> std::map<std::string, torch::Tensor> override;
    void set_parameters(const std::map<std::string, torch::Tensor>& params) override;
    auto to(torch::Device device) -> void override;

private:
    torch::Tensor weights_;
    torch::Tensor biases_;
    torch::Tensor input_cache_;
    torch::Tensor weights_gradient_;
    torch::Tensor biases_gradient_;
    float inertia_;
};