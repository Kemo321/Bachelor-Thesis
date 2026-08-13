#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <map>
#include <optional>
#include <string>

/**
 * @brief Fully connected layer using cuBLAS GEMM via dl::Tensor::matmul.
 *
 * Weights have shape [input_size, output_size]. Biases have shape [1, output_size].
 */
class FullyConnected : public Layer
{
public:
    /**
     * @brief Construct a fully connected layer.
     *
     * @param input_size Number of input features.
     * @param output_size Number of output features.
     * @param inertia_val Momentum factor used when accumulating gradients.
     */
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    FullyConnected(int input_size, int output_size, float inertia_val = 0.0F);

    /**
     * @brief Forward pass: Y = X W + b.
     * @param input_tensor Input activations with shape [batch, input_size].
     * @return Output activations with shape [batch, output_size].
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;

    /**
     * @brief Backward pass computing dX, dW, and db.
     * @param output_error_derivative Gradient wrt outputs with shape [batch, output_size].
     * @return Gradient wrt inputs with shape [batch, input_size].
     */
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;

    /**
     * @brief SGD parameter update.
     */
    void step() override;

    /**
     * @brief Retrieve learnable parameters.
     * @return Map with keys "weights" and "bias".
     */
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;

    /**
     * @brief Replace learnable parameters from an external source.
     */
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;

    /**
     * @brief FullyConnected parameters already reside on the GPU; CPU placement is rejected.
     */
    auto to(dl::Device device) -> void override;

private:
    dl::Tensor weights_;
    dl::Tensor biases_;
    std::optional<dl::Tensor> input_cache_;
    dl::Tensor weights_gradient_;
    dl::Tensor biases_gradient_;
    int input_size_;
    int output_size_;
    float inertia_;
};
