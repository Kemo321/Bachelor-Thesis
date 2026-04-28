#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>

class Conv2d : public Layer
{
public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val, float inertia_val = 0.0F);

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
    std::vector<int64_t> input_shape_cache_;
    int kernel_size_;
    int stride_;
    int padding_;
    float inertia_;
};