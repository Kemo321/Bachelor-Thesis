#include "DeepLearnLib/LeakyReLU.hpp"

LeakyReLU::LeakyReLU(float slope_val)
    : slope_(slope_val)
{
}

auto LeakyReLU::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    input_cache_ = input_tensor;

    return torch::where(input_tensor > 0, input_tensor, input_tensor * slope_);
}

auto LeakyReLU::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    torch::Tensor grad_activation = torch::where(
        input_cache_ > 0,
        torch::ones_like(input_cache_),
        torch::full_like(input_cache_, slope_));

    return output_error_derivative * grad_activation;
}
