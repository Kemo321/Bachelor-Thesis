#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

/**
 * Darknet `[local]` layer: per-spatial-location convolution (no weight sharing).
 *
 * Weight layout matches Darknet: `[locations, C_out, C_in, K, K]` with
 * `locations = out_h * out_w`. Biases are per output element `[C_out, out_h, out_w]`.
 */
class LocalLayer : public Layer
{
public:
    LocalLayer(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val, int in_height,
        int in_width);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    void clip_gradients(float abs_bound, cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

    [[nodiscard]] auto in_channels() const -> int
    {
        return in_channels_;
    }

    [[nodiscard]] auto out_channels() const -> int
    {
        return out_channels_;
    }

    [[nodiscard]] auto kernel_size() const -> int
    {
        return kernel_size_;
    }

    [[nodiscard]] auto out_height() const -> int
    {
        return out_height_;
    }

    [[nodiscard]] auto out_width() const -> int
    {
        return out_width_;
    }

    [[nodiscard]] auto locations() const -> int
    {
        return out_height_ * out_width_;
    }

private:
    dl::Tensor weights_;
    dl::Tensor biases_;
    dl::Tensor weights_gradient_;
    dl::Tensor biases_gradient_;
    std::optional<dl::Tensor> weights_velocity_;
    std::optional<dl::Tensor> biases_velocity_;
    std::optional<dl::Tensor> input_cache_;
    std::optional<dl::Tensor> output_cache_;
    std::optional<dl::Tensor> grad_input_cache_;
    bool input_cache_ready_ { false };

    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int in_height_;
    int in_width_;
    int out_height_;
    int out_width_;
};
