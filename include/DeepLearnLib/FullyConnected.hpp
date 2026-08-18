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
 * Backward uses logical GEMM transposes (CUBLAS_OP_T) instead of allocating
 * `input^T` / `W^T`, and SGD/weight-decay updates mutate buffers in place.
 */
class FullyConnected : public Layer
{
public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    FullyConnected(int input_size, int output_size, float inertia_val = 0.0F);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    void clip_gradients(float abs_bound, cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

    [[nodiscard]] auto input_size() const -> int
    {
        return input_size_;
    }

    [[nodiscard]] auto output_size() const -> int
    {
        return output_size_;
    }

private:
    dl::Tensor weights_;
    dl::Tensor biases_;
    std::optional<dl::Tensor> input_cache_;
    std::optional<dl::Tensor> output_cache_;
    std::optional<dl::Tensor> grad_input_cache_;
    bool input_cache_ready_ { false };
    dl::Tensor weights_gradient_;
    dl::Tensor biases_gradient_;
    std::optional<dl::Tensor> weights_velocity_;
    std::optional<dl::Tensor> biases_velocity_;
    int input_size_;
    int output_size_;
    float inertia_;
};
