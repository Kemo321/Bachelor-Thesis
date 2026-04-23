#include "DeepLearnLib/MaxPool2d.hpp"
#include <utility>

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
MaxPool2d::MaxPool2d(int kernel_size_val, int stride_val)
    : kernel_size_(kernel_size_val)
    , stride_(stride_val)
{
}

auto MaxPool2d::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    input_shape_cache_ = input_tensor.sizes().vec();

    auto output_tuple = torch::max_pool2d_with_indices(
        input_tensor, { kernel_size_, kernel_size_ }, { stride_, stride_ });

    indices_cache_ = std::get<1>(output_tuple);
    return std::get<0>(output_tuple);
}

auto MaxPool2d::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    return torch::nn::functional::max_unpool2d(
        output_error_derivative,
        indices_cache_,
        input_shape_cache_,
        torch::nn::functional::MaxUnpool2dFuncOptions()
            .kernel_size({ kernel_size_, kernel_size_ })
            .stride({ stride_, stride_ }));
}
