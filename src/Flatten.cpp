#include "DeepLearnLib/Flatten.hpp"

auto Flatten::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    input_shape_cache_ = input_tensor.sizes().vec();
    return input_tensor.view({ input_tensor.size(0), -1 });
}

auto Flatten::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    return output_error_derivative.view(input_shape_cache_);
}