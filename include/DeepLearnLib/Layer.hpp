#pragma once
#include <torch/torch.h>
#include <map>
#include <string>


class Layer {
public:
    float learning_rate = 0.001F;

    virtual ~Layer() = default;
    virtual void train() { is_training = true; }
    virtual void eval() { is_training = false; }

    [[nodiscard]] virtual auto forward(const torch::Tensor& input_tensor) -> torch::Tensor = 0;
    [[nodiscard]] virtual auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor = 0;

    virtual auto get_parameters() -> std::map<std::string, torch::Tensor> { return {}; }
    virtual void set_parameters(const std::map<std::string, torch::Tensor>& params) {}
    virtual auto to(torch::Device device) -> void { this->device = device; }

protected:
    bool is_training = true;
    torch::Device device = torch::kCPU;
};