#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>

/**
 * @brief Conv2d layer implementing a 2D convolution with manual parameter storage.
 *
 * This class implements a convolutional layer intended to be used with custom
 * forward/backward propagation logic (e.g., for educational or research code).
 * It stores weights and biases as tensors and exposes methods for forward pass,
 * backward pass (computing gradients), parameter updates (step), and device
 * transfer.
 */
class Conv2d : public Layer
{
public:
    /**
     * @brief Construct a Conv2d layer.
     *
     * @param in_channels Number of input channels (C_in).
     * @param out_channels Number of output channels / filters (C_out).
     * @param kernel_size Size of the square convolution kernel (K).
     * @param stride_val Stride for the convolution operation.
     * @param padding_val Padding applied to input on both sides.
     * @param inertia_val Momentum/inertia factor used in parameter updates.
     */
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val, float inertia_val = 0.0F);

    /**
     * @brief Forward pass of the convolutional layer.
     *
     * @param input_tensor Input tensor with shape [Batch, Channels_in, Height_in, Width_in].
     * @return Output tensor after convolution with shape [Batch, Channels_out, Height_out, Width_out].
     */
    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;

    /**
     * @brief Backward pass computing gradients w.r.t. inputs and parameters.
     *
     * @param output_error_derivative Gradient of the loss w.r.t. this layer's output.
     *        Expected shape: [Batch, Channels_out, Height_out, Width_out].
     * @return Gradient of the loss w.r.t. this layer's input. Shape: [Batch, Channels_in, Height_in, Width_in].
     * @note The implementation should follow the chain rule to propagate gradients
     *       to inputs and accumulate parameter gradients into weights_gradient_ and biases_gradient_.
     */
    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

    /**
     * @brief Update parameters (weights and biases) using accumulated gradients.
     *
     * This method applies the optimization step (including inertia/momentum if used).
     */
    void step() override;

    /**
     * @brief Retrieve current learnable parameters.
     *
     * @return Map of parameter name to tensor. Typical keys: "weights", "biases".
     */
    auto get_parameters() -> std::map<std::string, torch::Tensor> override;

    /**
     * @brief Replace current parameters from an external source.
     *
     * @param params Map containing tensors for parameters. Expected keys: "weights", "biases".
     */
    void set_parameters(const std::map<std::string, torch::Tensor>& params) override;

    /**
     * @brief Move internal tensors to the specified device.
     *
     * @param device Target torch device (e.g., torch::kCUDA or torch::kCPU).
     */
    auto to(torch::Device device) -> void override;

private:
    /** @brief Convolution weights tensor. Shape: [OutChannels, InChannels, Kernel, Kernel]. */
    torch::Tensor weights_;

    /** @brief Bias tensor. Shape: [OutChannels]. */
    torch::Tensor biases_;

    /** @brief Cached input tensor from the last forward pass. Shape: [Batch, InChannels, H, W]. */
    torch::Tensor input_cache_;

    /** @brief Accumulated gradients for weights, same shape as weights_. */
    torch::Tensor weights_gradient_;

    /** @brief Accumulated gradients for biases, same shape as biases_. */
    torch::Tensor biases_gradient_;

    /** @brief Cached input shape as {Batch, Channels, Height, Width}. */
    std::vector<int64_t> input_shape_cache_;

    /** @brief Kernel size (height == width). */
    int kernel_size_;

    /** @brief Stride value for convolution. */
    int stride_;

    /** @brief Padding value for convolution. */
    int padding_;

    /** @brief Inertia / momentum factor for parameter updates. */
    float inertia_;
};