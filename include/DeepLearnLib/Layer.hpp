#pragma once

#include <torch/torch.h>
#include <map>
#include <string>

/**
 * @brief Abstract base class representing a generic neural network layer.
 *
 * Provides a minimal interface for forward/backward passes, parameter
 * access, mode switching (train/eval) and device placement.
 */
class Layer {
public:
    /**
     * @brief Learning rate used by layers that implement parameter updates.
     * @note Public for quick experimentation; consider using accessors in
     *       larger codebases to avoid direct mutation.
     */
    float learning_rate = 0.001F;

    /**
     * @brief Virtual destructor.
     */
    virtual ~Layer() = default;

    /**
     * @brief Set layer to training mode.
     *
     * Some layers (e.g., dropout, batch-norm) alter behavior depending on
     * whether the model is training or evaluating.
     */
    virtual void train() { is_training_ = true; }

    /**
     * @brief Set layer to evaluation mode.
     */
    virtual void eval() { is_training_ = false; }

    /**
     * @brief Compute forward pass of the layer.
     * @param input_tensor Input tensor. Expected shape: [Batch, ...] where
     *        remaining dimensions depend on the concrete layer (e.g., for a
     *        convolutional layer: [Batch, Channels, Height, Width]).
     * @return Output tensor produced by the layer. Shape depends on the
     *         concrete implementation.
     */
    [[nodiscard]] virtual auto forward(const torch::Tensor& input_tensor) -> torch::Tensor = 0;

    /**
     * @brief Compute backward pass (gradient propagation) for the layer.
     * @param output_error_derivative Gradient of the loss w.r.t. the
     *        layer's output. Expected shape: same as the layer's forward
     *        output: [Batch, ...].
     * @return Gradient of the loss w.r.t. the layer's input. Shape matches
     *         the original forward input: [Batch, ...].
     * @note Implementations should follow the chain rule and ensure returned
     *       tensors are contiguous when necessary (use .contiguous()).
     */
    [[nodiscard]] virtual auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor = 0;

    /**
     * @brief Optional per-layer optimizer step.
     *
     * Layers that hold parameters and update them manually should override
     * this method. By default, no action is performed.
     */
    virtual void step() {}

    /**
     * @brief Retrieve named parameters owned by the layer.
     * @return Map from parameter name to tensor. Tensor shapes depend on the
     *         specific layer (e.g., weights: [Out, In], bias: [Out]).
     */
    virtual auto get_parameters() -> std::map<std::string, torch::Tensor> { return {}; }

    /**
     * @brief Set parameters by name.
     * @param params Map from parameter name to tensor. Shapes must match the
     *               layer's expected parameter shapes.
     */
    virtual void set_parameters(const std::map<std::string, torch::Tensor>& params) {}

    /**
     * @brief Move layer internal tensors to the given device.
     * @param device Target torch device (e.g., torch::kCUDA or torch::kCPU).
     */
    virtual auto to(torch::Device device) -> void { this->device_ = device; }

protected:
    /**
     * @brief Whether the layer is in training mode.
     * @note Use trailing underscore for private/protected member variables.
     */
    bool is_training_ = true;

    /**
     * @brief Device where the layer's tensors reside.
     */
    torch::Device device_ = torch::kCPU;
};
