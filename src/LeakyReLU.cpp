#include "DeepLearnLib/LeakyReLU.hpp"

/**
 * @brief Constructs a LeakyReLU activation layer.
 * 
 * @param slope_val The slope of the negative section.
 */
LeakyReLU::LeakyReLU(float slope_val)
    : slope_(slope_val)
{
}

/**
 * @brief Computes the forward pass of the LeakyReLU activation function.
 * 
 * @param input_tensor The input tensor of arbitrary shape.
 * @return The activated tensor of the same shape as input_tensor.
 */
auto LeakyReLU::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    input_cache_ = input_tensor;
    return torch::where(input_tensor > 0.0F, input_tensor, input_tensor * slope_);
}

/**
 * @brief Computes the backward pass of the LeakyReLU activation function.
 * 
 * @note Applies the chain rule using the cached input to compute the gradient.
 * 
 * @param output_error_derivative The derivative of the loss with respect to the output of this layer.
 * @return The derivative of the loss with respect to the input of this layer.
 */
auto LeakyReLU::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    auto gradient_activation = torch::where(input_cache_ > 0.0F, 1.0F, slope_);
    return output_error_derivative * gradient_activation;
}
