#include "DeepLearnLib/FullyConnected.hpp"
#include <cmath>

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FullyConnected::FullyConnected(int input_size, int output_size, float inertia_val)
    : inertia_(inertia_val)
{
    const float init_bound = std::sqrt(1.0F / static_cast<float>(input_size));
    constexpr float range_multiplier = 2.0F; // Fixes the magic number error

    weights_ = torch::rand({ input_size, output_size }) * range_multiplier * init_bound - init_bound;
    biases_ = torch::zeros({ 1, output_size });
    weights_gradient_ = torch::zeros_like(weights_);
}

auto FullyConnected::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    input_cache_ = input_tensor;
    return torch::matmul(input_tensor, weights_) + biases_;
}

auto FullyConnected::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    weights_gradient_ = torch::matmul(input_cache_.t(), output_error_derivative) + inertia_ * weights_gradient_;
    torch::Tensor biases_gradient = output_error_derivative.sum({ 0 }, /*keepdim=*/true);

    weights_ -= learning_rate * weights_gradient_;
    biases_ -= learning_rate * biases_gradient;

    return torch::matmul(output_error_derivative, weights_.t());
}
