#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>
#include <map>
#include <string>

class BatchNorm2d : public Layer
{
public:
    BatchNorm2d(int num_features, float eps = 1e-5F, float momentum = 0.1F);

    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;
    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

    auto get_parameters() -> std::map<std::string, torch::Tensor> override;
    void set_parameters(const std::map<std::string, torch::Tensor>& params) override;
    auto to(torch::Device target_device) -> void override;

private:
    int num_features_;
    float eps_;
    float momentum_bn_;

    torch::Tensor gamma_;
    torch::Tensor beta_;
    
    torch::Tensor gamma_grad_;
    torch::Tensor beta_grad_;

    torch::Tensor running_mean_;
    torch::Tensor running_var_;

    torch::Tensor x_hat_;
    torch::Tensor std_inv_;
};