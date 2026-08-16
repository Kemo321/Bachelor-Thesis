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
 * Spatial BatchNorm2d via cuDNN.
 *
 * Gamma, beta, running stats, and their gradients are GPU tensors [1, C, 1, 1],
 * matching cudnnDeriveBNTensorDescriptor for CUDNN_BATCHNORM_SPATIAL on NCHW inputs.
 */
class BatchNorm2d : public Layer
{
public:
    BatchNorm2d(int num_features, float eps = 1e-5F, float momentum = 0.1F);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    void clip_gradients(float abs_bound, cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

private:
    auto configure_descriptors(int batch, int channels, int height, int width, dl::Dtype dtype) -> void;

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
    bool descriptors_configured_ { false };
};
