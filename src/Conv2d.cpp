#include "DeepLearnLib/Conv2d.hpp"
#include <cmath>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val, float inertia_val)
    : stride_(stride_val)
    , padding_(padding_val)
    , inertia_(inertia_val)
{

    const float init_bound = std::sqrt(1.0F / static_cast<float>(in_channels * kernel_size * kernel_size));
    constexpr float range_multiplier = 2.0F;

    weights_ = torch::rand({ out_channels, in_channels, kernel_size, kernel_size }) * range_multiplier * init_bound - init_bound;
    biases_ = torch::zeros({ out_channels });
    weights_gradient_ = torch::zeros_like(weights_);
}

auto Conv2d::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    input_cache_ = input_tensor;
    return torch::conv2d(input_tensor, weights_, biases_, { stride_, stride_ }, { padding_, padding_ });
}

auto Conv2d::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    torch::Tensor biases_gradient = output_error_derivative.sum({ 0, 2, 3 });

    torch::Tensor current_weights_gradient = torch::nn::grad::conv2d_weight(
        input_cache_, weights_.sizes(), output_error_derivative, { stride_, stride_ }, { padding_, padding_ });

    weights_gradient_ = current_weights_gradient + inertia_ * weights_gradient_;

    torch::Tensor grad_input = torch::conv_transpose2d(
        output_error_derivative, weights_, {}, { stride_, stride_ }, { padding_, padding_ });

    weights_ -= learning_rate * weights_gradient_;
    biases_ -= learning_rate * biases_gradient;

    return grad_input;
}
