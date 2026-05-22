#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <map>
#include <string>
#include <torch/torch.h>

/**
 * @brief Fully connected (dense) linear layer implementing manual forward/backward.
 *
 * This layer stores weights and biases and exposes explicit forward, backward
 * and parameter update (step) operations. It is intended for use in a
 * custom training loop where gradients are computed via the backward method
 * and applied in step().
 */
class FullyConnected : public Layer
{
public:
    /**
     * @brief Construct a fully connected layer.
     *
     * @param inputSize Number of input features (D_in).
     * @param outputSize Number of output features (D_out).
     * @param inertiaVal Momentum/inertia factor used in parameter updates (default 0.0F).
     */
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    FullyConnected(int inputSize, int outputSize, float inertiaVal = 0.0F);

    /**
     * @brief Forward pass for the fully connected layer.
     *
     * @param inputTensor Input activation tensor with shape [batch, inputSize].
     *                    Prefer contiguous memory layout; implementations may
     *                    call .contiguous() internally to avoid striding issues.
     * @return Output activation tensor with shape [batch, outputSize].
     */
    [[nodiscard]] auto forward(const torch::Tensor& inputTensor) -> torch::Tensor override;

    /**
     * @brief Backward pass computing derivative wrt inputs.
     *
     * @param outputErrorDerivative Gradient of loss wrt this layer's outputs with shape [batch, outputSize].
     * @return Gradient of loss wrt this layer's inputs with shape [batch, inputSize].
     * @note This method should populate internal weight/bias gradient members used by step().
     */
    [[nodiscard]] auto backward(const torch::Tensor& outputErrorDerivative) -> torch::Tensor override;

    /**
     * @brief Apply accumulated gradients to parameters (weights and biases).
     *
     * This method should update internal parameters according to the chosen
     * optimizer scheme (e.g. simple SGD with optional inertia/momentum).
     */
    void step() override;

    /**
     * @brief Retrieve layer parameters.
     *
     * @return Map with keys typically "weights" and "biases" and their tensors.
     */
    auto get_parameters() -> std::map<std::string, torch::Tensor> override;

    /**
     * @brief Replace layer parameters from provided map.
     *
     * @param params Map containing tensors for keys such as "weights" and "biases".
     */
    void set_parameters(const std::map<std::string, torch::Tensor>& params) override;

    /**
     * @brief Move internal tensors to the specified device.
     *
     * @param device Target torch device (e.g. torch::kCUDA or torch::kCPU).
     */
    auto to(torch::Device device) -> void override;

private:
    /**
     * @brief Weight matrix with shape [inputSize, outputSize].
     */
    torch::Tensor weights_;

    /**
     * @brief Bias vector with shape [outputSize].
     */
    torch::Tensor biases_;

    /**
     * @brief Cached input activations from last forward pass with shape [batch, inputSize].
     */
    torch::Tensor input_cache_;

    /**
     * @brief Gradient accumulator for weights, same shape as weights_.
     */
    torch::Tensor weights_gradient_;

    /**
     * @brief Gradient accumulator for biases, same shape as biases_.
     */
    torch::Tensor biases_gradient_;

    /**
     * @brief Inertia / momentum coefficient applied during parameter updates.
     */
    float inertia_;
};
