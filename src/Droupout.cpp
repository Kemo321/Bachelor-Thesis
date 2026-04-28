#include "DeepLearnLib/Dropout.hpp"

Dropout::Dropout(float probability)
    : probability_(probability)
{
}

auto Dropout::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    if (is_training)
    {
        mask_ = (torch::rand(input_tensor.sizes(), input_tensor.options()) > probability_).to(torch::kFloat32);

        mask_ = mask_ / (1.0F - probability_);

        return input_tensor * mask_;
    }
    
    return input_tensor;
}

auto Dropout::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    return output_error_derivative * mask_;
}