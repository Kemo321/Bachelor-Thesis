#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>
#include <map>
#include <string>

/**
 * @brief Batch Normalization 2D layer.
 */
class BatchNorm2d : public Layer
{
public:
    /**
     * @brief Constructs a BatchNorm2d layer.
     * @param numFeatures The number of features or channels.
     * @param eps A value added to the denominator for numerical stability.
     * @param momentum The value used for the running_mean and running_var computation.
     */
    BatchNorm2d(int numFeatures, float eps = 1e-5F, float momentum = 0.1F);

    /**
     * @brief Performs the forward pass computation.
     * @param inputTensor Input tensor of shape [Batch, Channels, Height, Width].
     * @return Output tensor of shape [Batch, Channels, Height, Width].
     */
    [[nodiscard]] auto forward(const torch::Tensor& inputTensor) -> torch::Tensor override;

    /**
     * @brief Performs the backward pass computation.
     * @param outputErrorDerivative Error derivative from the next layer of shape [Batch, Channels, Height, Width].
     * @return Input error derivative of shape [Batch, Channels, Height, Width].
     * @note Chain rule derivative for batch normalization is applied here to calculate gradients for inputs, gamma, and beta.
     */
    [[nodiscard]] auto backward(const torch::Tensor& outputErrorDerivative) -> torch::Tensor override;

    /**
     * @brief Updates the learnable parameters (gamma and beta) using the computed gradients.
     */
    void step() override;

    /**
     * @brief Retrieves the learnable parameters of the layer.
     * @return A map of parameter names to their corresponding tensors.
     */
    auto get_parameters() -> std::map<std::string, torch::Tensor> override;

    /**
     * @brief Sets the learnable parameters of the layer.
     * @param params A map of parameter names to their corresponding tensors.
     */
    void set_parameters(const std::map<std::string, torch::Tensor>& params) override;

    /**
     * @brief Moves the layer's parameters and buffers to the specified device.
     * @param targetDevice The target PyTorch device.
     */
    auto to(torch::Device targetDevice) -> void override;

private:
    int num_features_;
    float eps_;
    float momentum_bn_;

    torch::Tensor gamma_;
    torch::Tensor beta_;
    
    torch::Tensor gamma_grad_;
    torch::Tensor beta_grad_;

    torch::Tensor running_mean_;
    torch::Tensor running_var_;

    torch::Tensor x_hat_;
    torch::Tensor std_inv_;
};