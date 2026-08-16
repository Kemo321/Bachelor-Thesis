#pragma once

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

/**
 * Fused Conv2d -> BatchNorm2d -> LeakyReLU block.
 *
 * Convolution and bias are launched with `cudnnConvolutionBiasActivationForward`
 * (IDENTITY activation so BatchNorm still sees a linear pre-activation).
 * BatchNorm affine + LeakyReLU then run in a single elementwise CUDA kernel,
 * avoiding the extra global-memory round trip of a standalone activation.
 */
class FusedCBR2d : public Layer
{
public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    FusedCBR2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val,
        float leaky_slope = 0.1F, float bn_eps = 1e-5F, float bn_momentum = 0.1F);

    void train() override;
    void eval() override;

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    void clip_gradients(float abs_bound, cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

    [[nodiscard]] auto leaky_slope() const -> float
    {
        return leaky_slope_;
    }

private:
    auto configure_bn_descriptors(const dl::Tensor& conv_output) -> void;
    auto apply_bn_leaky(const dl::Tensor& conv_output, cudaStream_t stream) -> dl::Tensor;

    Conv2d conv_;
    float leaky_slope_;
    float bn_eps_;
    float bn_momentum_;
    int out_channels_;

    dl::Tensor gamma_;
    dl::Tensor beta_;
    dl::Tensor gamma_grad_;
    dl::Tensor beta_grad_;
    dl::Tensor running_mean_;
    dl::Tensor running_var_;
    dl::Tensor batch_var_;
    dl::Tensor save_mean_;
    dl::Tensor save_inv_var_;

    std::optional<dl::Tensor> bn_input_cache_;
    std::optional<dl::Tensor> fused_output_cache_;
    std::vector<int> bn_shape_cache_;

    dl::CudnnTensorDescriptor x_desc_;
    dl::CudnnTensorDescriptor bn_desc_;
    bool bn_descriptors_configured_ { false };
};
