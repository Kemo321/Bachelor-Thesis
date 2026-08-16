#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <map>
#include <optional>
#include <string>

/**
 * Fully connected layer using cuBLAS GEMM via dl::Tensor::matmul.
 *
 * Weights have shape [input_size, output_size]. Biases have shape [1, output_size].
 */
class FullyConnected : public Layer
{
public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    FullyConnected(int input_size, int output_size, float inertia_val = 0.0F);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;
    void step() override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
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
