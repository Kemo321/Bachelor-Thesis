#include "DeepLearnLib/Dropout.hpp"

/**
 * @brief Constructs a Dropout layer.
 * @param probability The probability of an element to be zeroed.
 */
Dropout::Dropout(float probability)
    : probability_(probability)
{
}

/**
 * @brief Performs the forward pass of the Dropout layer.
 * @param input_tensor The input tensor.
 * @return The output tensor after applying dropout.
 */
auto Dropout::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    if (is_training_)
    {
        mask_ = (torch::rand(input_tensor.sizes(), input_tensor.options()) > probability_).to(torch::kFloat32);

        mask_ = mask_ / (1.0F - probability_);

        return input_tensor * mask_;
    }
    
    return input_tensor;
}

/**
 * @brief Performs the backward pass of the Dropout layer.
 * @param output_error_derivative The derivative of the error with respect to the output.
 * @return The derivative of the error with respect to the input.
 * @note Computes the gradient by passing it only through the active (non-dropped) nodes.
 */
auto Dropout::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    return output_error_derivative * mask_;
}