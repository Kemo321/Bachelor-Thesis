#include "DeepLearnLib/MaxPool2d.hpp"
#include <tuple>
#include <utility>

/**
 * @brief Constructs a MaxPool2d layer with specified kernel size and stride.
 * @param kernel_size_val The size of the pooling kernel (kernel_size_val x kernel_size_val).
 * @param stride_val The stride of the pooling operation.
 */
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
MaxPool2d::MaxPool2d(int kernel_size_val, int stride_val)
    : kernel_size_(kernel_size_val)
    , stride_(stride_val)
{
}

/**
 * @brief Performs forward pass of 2D max pooling operation.
 * 
 * Applies 2D max pooling using torch::nn::functional::unfold and fold operations.
 * This implementation extracts patches, finds maximum values per patch, and reconstructs
 * the output tensor. Indices of maximum values are cached for backward pass.
 * 
 * @param input_tensor Input tensor of shape [Batch, Channels, Height, Width].
 * @return Output tensor of shape [Batch, Channels, OutputHeight, OutputWidth],
 *         where OutputHeight = (Height - kernel_size) / stride + 1,
 *         and OutputWidth = (Width - kernel_size) / stride + 1.
 * 
 * @note The maximum value indices are cached in indices_cache_ for use in the backward pass.
 *       The input tensor shape is cached in input_shape_cache_ to reconstruct gradients.
 */
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

    return max_values.contiguous().view({ batch_size, channels_count, output_height, output_width });
}

/**
 * @brief Computes the gradients for the backward pass of the MaxPool2d layer.
 * 
 * @param output_error_derivative Gradient of the loss with respect to the layer's output [Batch, Channels, OutputHeight, OutputWidth].
 * @return Gradient of the loss with respect to the layer's input [Batch, Channels, Height, Width].
 * 
 * @note Uses the indices cached during the forward pass to scatter the gradients back to the corresponding maximum value locations.
 */
auto MaxPool2d::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    int64_t batch_size = output_error_derivative.size(0);
    int64_t channels_count = output_error_derivative.size(1);
    int64_t patch_area = static_cast<int64_t>(kernel_size_) * kernel_size_;
    int64_t patches_count = indices_cache_.size(2);

    torch::Tensor flat_gradient = output_error_derivative.contiguous().view({ batch_size, channels_count, 1, patches_count });
    
    torch::Tensor gradient_unfolded = torch::zeros(
        { batch_size, channels_count, patch_area, patches_count }, 
        output_error_derivative.options());

    constexpr int scatter_dimension = 2;
    torch::Tensor scatter_indices = indices_cache_.unsqueeze(scatter_dimension);

    gradient_unfolded.scatter_(scatter_dimension, scatter_indices, flat_gradient);

    gradient_unfolded = gradient_unfolded.view({ batch_size, channels_count * patch_area, patches_count });

    auto folded_grad = torch::nn::functional::fold(
        gradient_unfolded,
        torch::nn::functional::FoldFuncOptions({ input_shape_cache_[2], input_shape_cache_[3] }, { kernel_size_, kernel_size_ })
            .stride({ stride_, stride_ }));

    indices_cache_ = torch::Tensor();

    return folded_grad;
}
