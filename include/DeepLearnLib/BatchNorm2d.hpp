#pragma once

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cudnn.h>
#include <map>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Batch Normalization 2D layer implemented with cuDNN spatial BN.
 *
 * Scale (gamma), bias (beta), running statistics, and their gradients are stored
 * as GPU tensors with shape [1, C, 1, 1], matching cudnnDeriveBNTensorDescriptor
 * for CUDNN_BATCHNORM_SPATIAL on NCHW inputs.
 */
class BatchNorm2d : public Layer
{
public:
    /**
     * @brief Constructs a BatchNorm2d layer.
     * @param num_features Number of features or channels.
     * @param eps Value added to the denominator for numerical stability.
     * @param momentum Exponential average factor used to update running statistics.
     */
    BatchNorm2d(int num_features, float eps = 1e-5F, float momentum = 0.1F);

    /**
     * @brief Performs the forward pass computation.
     * @param input_tensor Input tensor of shape [Batch, Channels, Height, Width].
     * @return Output tensor of shape [Batch, Channels, Height, Width].
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;

    /**
     * @brief Performs the backward pass computation.
     * @param output_error_derivative Error derivative from the next layer of shape [Batch, Channels, Height, Width].
     * @return Input error derivative of shape [Batch, Channels, Height, Width].
     */
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;

    /**
     * @brief Updates the learnable parameters (gamma and beta) using the computed gradients.
     */
    void step() override;

    /**
     * @brief Retrieves the learnable parameters and running statistics of the layer.
     */
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;

    /**
     * @brief Sets the learnable parameters and running statistics of the layer.
     */
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;

    /**
     * @brief BatchNorm2d parameters already reside on the GPU; CPU placement is rejected.
     */
    auto to(dl::Device device) -> void override;

private:
    auto configure_descriptors(int batch, int channels, int height, int width) -> void;

    int num_features_;
    float eps_;
    float momentum_bn_;

    dl::Tensor gamma_;
    dl::Tensor beta_;
    dl::Tensor gamma_grad_;
    dl::Tensor beta_grad_;
    dl::Tensor running_mean_;
    dl::Tensor running_var_;
    dl::Tensor save_mean_;
    dl::Tensor save_inv_var_;

    std::optional<dl::Tensor> input_cache_;
    std::vector<int> input_shape_cache_;

    dl::CudnnTensorDescriptor x_desc_;
    dl::CudnnTensorDescriptor bn_desc_;
    bool descriptors_configured_{ false };
};
