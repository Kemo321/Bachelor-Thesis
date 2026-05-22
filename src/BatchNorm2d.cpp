#include "DeepLearnLib/BatchNorm2d.hpp"

/**
 * @brief Constructs a 2D Batch Normalization layer.
 * 
 * @param num_features Number of features/channels in the input tensor.
 * @param eps A value added to the denominator for numerical stability.
 * @param momentum The value used for the running_mean and running_var computation.
 */
BatchNorm2d::BatchNorm2d(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_bn_(momentum)
{
    gamma_ = torch::ones({1, num_features, 1, 1});
    beta_ = torch::zeros({1, num_features, 1, 1});
    
    gamma_grad_ = torch::zeros_like(gamma_);
    beta_grad_ = torch::zeros_like(beta_);

    running_mean_ = torch::zeros({1, num_features, 1, 1});
    running_var_ = torch::ones({1, num_features, 1, 1});
}

/**
 * @brief Performs the forward pass of the batch normalization.
 * 
 * @param input_tensor The input tensor [Batch, Channels, Height, Width].
 * @return torch::Tensor The normalized output tensor [Batch, Channels, Height, Width].
 */
auto BatchNorm2d::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    if (is_training_)
    {
        auto mean = input_tensor.mean({0, 2, 3}, true);
        auto var = input_tensor.var({0, 2, 3}, false, true);

        std_inv_ = 1.0F / torch::sqrt(var + eps_);
        x_hat_ = (input_tensor - mean) * std_inv_;

        running_mean_ = (1.0F - momentum_bn_) * running_mean_ + momentum_bn_ * mean.detach();
        
        int64_t numElements = input_tensor.size(0) * input_tensor.size(2) * input_tensor.size(3);
        float adjust = static_cast<float>(numElements) / static_cast<float>(numElements - 1);
        running_var_ = (1.0F - momentum_bn_) * running_var_ + momentum_bn_ * var.detach() * adjust;

        return gamma_ * x_hat_ + beta_;
    }
    
    auto xHat = (input_tensor - running_mean_) / torch::sqrt(running_var_ + eps_);
    return gamma_ * xHat + beta_;
}

/**
 * @brief Performs the backward pass, computing gradients for inputs and parameters.
 * 
 * @note Applies the chain rule to backpropagate through the normalization steps.
 * 
 * @param output_error_derivative The gradient of the loss with respect to the output [Batch, Channels, Height, Width].
 * @return torch::Tensor The gradient of the loss with respect to the input [Batch, Channels, Height, Width].
 */
auto BatchNorm2d::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor {
    if (!is_training_ || !x_hat_.defined()) {
        return output_error_derivative; 
    }

    gamma_grad_ = (output_error_derivative * x_hat_).sum({0, 2, 3}, true);
    beta_grad_  = output_error_derivative.sum({0, 2, 3}, true);

    constexpr float weightDecay = 0.0005F;
    gamma_grad_ += weightDecay * gamma_;
    beta_grad_  += weightDecay * beta_;

    auto gradXHat = output_error_derivative * gamma_;
    auto meanGradXHat = gradXHat.mean({0, 2, 3}, true);
    auto varTerm = (gradXHat * x_hat_).mean({0, 2, 3}, true);
    auto gradInput = std_inv_ * (gradXHat - meanGradXHat - x_hat_ * varTerm);

    x_hat_ = torch::Tensor();
    std_inv_ = torch::Tensor();
    return gradInput; // return dx
}

/**
 * @brief Updates the gamma and beta parameters using their computed gradients.
 */
void BatchNorm2d::step() {
    gamma_ -= learning_rate * gamma_grad_;
    beta_  -= learning_rate * beta_grad_;
}

/**
 * @brief Moves parameters and buffers to the specified target device.
 * 
 * @param target_device The device to move the tensors to (e.g., CPU, CUDA).
 */
auto BatchNorm2d::to(torch::Device target_device) -> void
{
    Layer::to(target_device);
    gamma_ = gamma_.to(target_device);
    beta_ = beta_.to(target_device);
    gamma_grad_ = gamma_grad_.to(target_device);
    beta_grad_ = beta_grad_.to(target_device);
    running_mean_ = running_mean_.to(target_device);
    running_var_ = running_var_.to(target_device);
    
    if (x_hat_.defined()) {
        x_hat_ = x_hat_.to(target_device);
    }
    if (std_inv_.defined()) {
        std_inv_ = std_inv_.to(target_device);
    }
}

/**
 * @brief Gets a map of the layer's parameters and buffers.
 * 
 * @return std::map<std::string, torch::Tensor> A dictionary containing parameter tensors.
 */
auto BatchNorm2d::get_parameters() -> std::map<std::string, torch::Tensor>
{
    return {
        {"gamma", gamma_},
        {"beta", beta_},
        {"running_mean", running_mean_},
        {"running_var", running_var_}
    };
}

/**
 * @brief Sets the layer's parameters and buffers from a map.
 * 
 * @param params A dictionary containing parameter tensors.
 */
void BatchNorm2d::set_parameters(const std::map<std::string, torch::Tensor>& params)
{
    gamma_ = params.at("gamma");
    beta_ = params.at("beta");
    running_mean_ = params.at("running_mean");
    running_var_ = params.at("running_var");
}
