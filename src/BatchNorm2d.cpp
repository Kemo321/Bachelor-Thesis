#include "DeepLearnLib/BatchNorm2d.hpp"

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

auto BatchNorm2d::forward(const torch::Tensor& input_tensor) -> torch::Tensor
{
    if (is_training)
    {
        auto mean = input_tensor.mean({0, 2, 3}, true);
        auto var = input_tensor.var({0, 2, 3}, false, true);

        std_inv_ = 1.0F / torch::sqrt(var + eps_);
        x_hat_ = (input_tensor - mean) * std_inv_;

        running_mean_ = (1.0F - momentum_bn_) * running_mean_ + momentum_bn_ * mean.detach();
        
        int64_t n = input_tensor.size(0) * input_tensor.size(2) * input_tensor.size(3);
        float adjust = static_cast<float>(n) / static_cast<float>(n - 1);
        running_var_ = (1.0F - momentum_bn_) * running_var_ + momentum_bn_ * var.detach() * adjust;

        return gamma_ * x_hat_ + beta_;
    }
    
    auto x_hat = (input_tensor - running_mean_) / torch::sqrt(running_var_ + eps_);
    return gamma_ * x_hat + beta_;
}

auto BatchNorm2d::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor
{
    if (!is_training || !x_hat_.defined()) {
        return output_error_derivative; 
    }

    gamma_grad_ = (output_error_derivative * x_hat_).sum({0,2,3}, true);
    beta_grad_  = output_error_derivative.sum({0,2,3}, true);

    constexpr float weight_decay = 0.0005F;
    gamma_grad_ += weight_decay * gamma_;
    beta_grad_  += weight_decay * beta_;

    gamma_ -= learning_rate * gamma_grad_;
    beta_  -= learning_rate * beta_grad_;

    auto dx_hat = output_error_derivative * gamma_;

    auto mean_dx = dx_hat.mean({0,2,3}, true);
    auto var_term = (dx_hat * x_hat_).mean({0,2,3}, true);

    auto dx = std_inv_ * (dx_hat - mean_dx - x_hat_ * var_term);

    return dx;
}

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

auto BatchNorm2d::get_parameters() -> std::map<std::string, torch::Tensor>
{
    return {
        {"gamma", gamma_},
        {"beta", beta_},
        {"running_mean", running_mean_},
        {"running_var", running_var_}
    };
}

void BatchNorm2d::set_parameters(const std::map<std::string, torch::Tensor>& params)
{
    gamma_ = params.at("gamma");
    beta_ = params.at("beta");
    running_mean_ = params.at("running_mean");
    running_var_ = params.at("running_var");
}