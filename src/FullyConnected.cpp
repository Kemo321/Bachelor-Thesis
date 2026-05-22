#include "DeepLearnLib/FullyConnected.hpp"
#include <cmath>

/**
 * @brief Constructs a FullyConnected (Dense) layer.
 * 
 * Initializes weights and biases using internal PyTorch uniform initializers.
 * 
 * @param input_size Number of input features.
 * @param output_size Number of output features.
 * @param inertia_val Momentum/inertia value for gradient updates.
 */
FullyConnected::FullyConnected(int input_size, int output_size, float inertia_val)
    : inertia_(inertia_val)
{
    weights_ = torch::empty({ input_size, output_size });
    biases_ = torch::empty({ 1, output_size });

    float k = 1.0F / static_cast<float>(input_size);
    float bound = std::sqrt(k);

    torch::nn::init::uniform_(weights_, -bound, bound);
    torch::nn::init::uniform_(biases_, -bound, bound);
    
    weights_gradient_ = torch::zeros_like(weights_);
    biases_gradient_ = torch::zeros_like(biases_);
}

/**
 * @brief Computes the forward pass for the layer.
 * 
 * @param input_tensor Tensor of shape [Batch, input_size].
 * @return torch::Tensor Output tensor of shape [Batch, output_size].
 */
auto FullyConnected::forward(const torch::Tensor& input_tensor) -> torch::Tensor {
    input_cache_ = input_tensor;
    return torch::matmul(input_tensor, weights_) + biases_;
}

/**
 * @brief Computes the backward pass and accumulates gradients.
 * 
 * @note Applies the chain rule to accumulate weight and bias gradients, considering L2 weight decay.
 * 
 * @param output_error_derivative Gradient of the loss with respect to the layer's output, shape [Batch, output_size].
 * @return torch::Tensor Gradient of the loss with respect to the layer's input, shape [Batch, input_size].
 */
auto FullyConnected::backward(const torch::Tensor& output_error_derivative) -> torch::Tensor {
    auto cur_weights_grad = torch::matmul(input_cache_.t(), output_error_derivative);
    auto cur_biases_grad = output_error_derivative.sum(0, true);

    constexpr float weight_decay = 0.0005F;
    cur_weights_grad += weight_decay * weights_;
    cur_biases_grad += weight_decay * biases_;

    weights_gradient_ = cur_weights_grad + inertia_ * weights_gradient_;
    biases_gradient_ = cur_biases_grad + inertia_ * biases_gradient_;

    auto grad_input = torch::matmul(output_error_derivative, weights_.t());
    input_cache_ = torch::Tensor();
    return grad_input; // Ensure grad_input is passed down correctly
}

/**
 * @brief Updates the weights and biases using accumulated gradients.
 */
void FullyConnected::step() {
    weights_ -= learning_rate * weights_gradient_;
    biases_ -= learning_rate * biases_gradient_; 
}

/**
 * @brief Moves the layer's tensors to the specified hardware device.
 * 
 * @param device The target torch::Device (e.g., CPU, CUDA).
 */
auto FullyConnected::to(torch::Device device) -> void {
    weights_ = weights_.to(device);
    biases_ = biases_.to(device);
    weights_gradient_ = weights_gradient_.to(device);
    biases_gradient_ = biases_gradient_.to(device);
}

/**
 * @brief Retrieves the parameters of the layer.
 * 
 * @return std::map<std::string, torch::Tensor> Layer parameters formatted as a map of tensors.
 */
auto FullyConnected::get_parameters() -> std::map<std::string, torch::Tensor> {
    return {{"weights", weights_}, {"bias", biases_}};
}

/**
 * @brief Sets the parameters of the layer from a given map.
 * 
 * @param params Input map containing updated "weights" and "bias" tensors.
 */
auto FullyConnected::set_parameters(const std::map<std::string, torch::Tensor>& params) -> void {
    weights_ = params.at("weights");
    biases_ = params.at("bias");
}