#include "DeepLearnLib/MaxPool2d.hpp"
#include <tuple>
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
    
    int64_t batch_size = input_tensor.size(0);
    int64_t channels_count = input_tensor.size(1);
    int64_t input_height = input_tensor.size(2);
    int64_t input_width = input_tensor.size(3);

    torch::Tensor unfolded_input = torch::nn::functional::unfold(
        input_tensor,
        torch::nn::functional::UnfoldFuncOptions({ kernel_size_, kernel_size_ }).stride({ stride_, stride_ }));

    int64_t patch_area = static_cast<int64_t>(kernel_size_) * kernel_size_;
    int64_t patches_count = unfolded_input.size(2);

    torch::Tensor reshaped_unfolded = unfolded_input.view({ batch_size, channels_count, patch_area, patches_count });

    constexpr int reduction_dimension = 2;
    auto max_tuple = torch::max(reshaped_unfolded, reduction_dimension, false);

    torch::Tensor max_values = std::get<0>(max_tuple);
    indices_cache_ = std::get<1>(max_tuple);

    int64_t output_height = (input_height - kernel_size_) / stride_ + 1;
    int64_t output_width = (input_width - kernel_size_) / stride_ + 1;

    return max_values.view({ batch_size, channels_count, output_height, output_width });
}

auto MaxPool2d::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    int64_t batch_size = output_error_derivative.size(0);
    int64_t channels_count = output_error_derivative.size(1);
    int64_t patch_area = static_cast<int64_t>(kernel_size_) * kernel_size_;
    int64_t patches_count = indices_cache_.size(2);

    torch::Tensor flat_gradient = output_error_derivative.view({ batch_size, channels_count, 1, patches_count });
    
    torch::Tensor gradient_unfolded = torch::zeros(
        { batch_size, channels_count, patch_area, patches_count }, 
        output_error_derivative.options());

    constexpr int scatter_dimension = 2;
    torch::Tensor scatter_indices = indices_cache_.unsqueeze(scatter_dimension);

    gradient_unfolded.scatter_(scatter_dimension, scatter_indices, flat_gradient);

    gradient_unfolded = gradient_unfolded.view({ batch_size, channels_count * patch_area, patches_count });

    return torch::nn::functional::fold(
        gradient_unfolded,
        torch::nn::functional::FoldFuncOptions({ input_shape_cache_[2], input_shape_cache_[3] }, { kernel_size_, kernel_size_ })
            .stride({ stride_, stride_ }));
}