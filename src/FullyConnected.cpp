#include "DeepLearnLib/FullyConnected.hpp"
#include <cmath>

FullyConnected::FullyConnected(int input_size, int output_size, float inertia_val)
    : inertia_(inertia_val)
{
    const float std_dev = std::sqrt(2.0F / static_cast<float>(input_size));
    weights_ = torch::randn({ input_size, output_size }) * std_dev;
    biases_ = torch::zeros({ 1, output_size });
    
    weights_gradient_ = torch::zeros_like(weights_);
    biases_gradient_ = torch::zeros_like(biases_);
}

auto FullyConnected::forward(const torch::Tensor& input_tensor) -> torch::Tensor {
    input_cache_ = input_tensor;
    return torch::matmul(input_tensor, weights_) + biases_;
}

auto FullyConnected::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor {
    auto cur_weights_grad = torch::matmul(input_cache_.t(), output_error_derivative);
    auto cur_biases_grad = output_error_derivative.sum(0, true);

    constexpr float weight_decay = 0.0005F;
    cur_weights_grad += weight_decay * weights_;
    cur_biases_grad += weight_decay * biases_;

    weights_gradient_ = cur_weights_grad + inertia_ * weights_gradient_;
    biases_gradient_ = cur_biases_grad + inertia_ * biases_gradient_;

    weights_ -= learning_rate * weights_gradient_;
    biases_ -= learning_rate * biases_gradient_; 
    
    return torch::matmul(output_error_derivative, weights_.t());
}

auto FullyConnected::to(torch::Device device) -> void {
    weights_ = weights_.to(device);
    biases_ = biases_.to(device);
    weights_gradient_ = weights_gradient_.to(device);
    biases_gradient_ = biases_gradient_.to(device);
}

auto FullyConnected::get_parameters() -> std::map<std::string, torch::Tensor> {
    return {{"weights", weights_}, {"bias", biases_}};
}

auto FullyConnected::set_parameters(const std::map<std::string, torch::Tensor>& params) -> void {
    weights_ = params.at("weights");
    biases_ = params.at("bias");
}