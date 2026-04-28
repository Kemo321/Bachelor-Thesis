#include "DeepLearnLib/Conv2d.hpp"
#include <cmath>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val, float inertia_val)
    : stride_(stride_val), padding_(padding_val), inertia_(inertia_val), kernel_size_(kernel_size)
{
    const float std_dev = std::sqrt(2.0F / static_cast<float>(in_channels * kernel_size * kernel_size));
    weights_ = torch::randn({ out_channels, in_channels, kernel_size, kernel_size }) * std_dev;
    biases_ = torch::zeros({ out_channels });
    
    weights_gradient_ = torch::zeros_like(weights_);
    biases_gradient_ = torch::zeros_like(biases_); // Bufor momentum
}

auto Conv2d::forward(const torch::Tensor& input_tensor) -> torch::Tensor {
    input_shape_cache_ = input_tensor.sizes().vec();
    input_cache_ = torch::nn::functional::unfold(input_tensor, 
        torch::nn::functional::UnfoldFuncOptions({kernel_size_, kernel_size_}).stride({stride_, stride_}).padding({padding_, padding_}));
    
    auto reshaped_weights = weights_.view({weights_.size(0), -1});
    auto output = torch::matmul(reshaped_weights, input_cache_);
    output += biases_.view({1, -1, 1});

    int h_out = (input_shape_cache_[2] + 2 * padding_ - kernel_size_) / stride_ + 1;
    int w_out = (input_shape_cache_[3] + 2 * padding_ - kernel_size_) / stride_ + 1;
    return output.view({input_shape_cache_[0], weights_.size(0), h_out, w_out});
}

auto Conv2d::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor {
    auto reshaped_error = output_error_derivative.view({output_error_derivative.size(0), output_error_derivative.size(1), -1});
    
    auto cur_biases_grad = reshaped_error.sum(std::vector<int64_t>{0, 2});
    auto cur_weights_grad = torch::bmm(reshaped_error, input_cache_.transpose(1, 2)).sum(0).view_as(weights_);
    
    weights_gradient_ = cur_weights_grad + inertia_ * weights_gradient_;
    biases_gradient_ = cur_biases_grad + inertia_ * biases_gradient_;
    
    weights_ -= learning_rate * weights_gradient_;
    biases_ -= learning_rate * biases_gradient_;
    
    auto reshaped_weights = weights_.view({weights_.size(0), -1});
    auto grad_input_unfolded = torch::matmul(reshaped_weights.t(), reshaped_error);
    
    return torch::nn::functional::fold(grad_input_unfolded,
        torch::nn::functional::FoldFuncOptions({input_shape_cache_[2], input_shape_cache_[3]}, {kernel_size_, kernel_size_})
        .stride({stride_, stride_}).padding({padding_, padding_}));
}

auto Conv2d::to(torch::Device device) -> void {
    weights_ = weights_.to(device);
    biases_ = biases_.to(device);
    weights_gradient_ = weights_gradient_.to(device);
    biases_gradient_ = biases_gradient_.to(device);
}

auto Conv2d::get_parameters() -> std::map<std::string, torch::Tensor> {
    return {{"weights", weights_}, {"bias", biases_}};
}

auto Conv2d::set_parameters(const std::map<std::string, torch::Tensor>& params) -> void {
    weights_ = params.at("weights");
    biases_ = params.at("bias");
}