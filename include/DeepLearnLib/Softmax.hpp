#pragma once

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <optional>
#include <vector>

/**
 * @brief Softmax layer via cuDNN (accurate algorithm).
 *
 * Rank-2 inputs [N, C] softmax over C. Rank-4 NCHW inputs softmax over the
 * channel dimension at each spatial location (CUDNN_SOFTMAX_MODE_CHANNEL).
 */
class Softmax : public Layer
{
public:
    Softmax() = default;

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;

private:
    auto configure_descriptor(const dl::Tensor& tensor) -> void;

    std::optional<dl::Tensor> output_cache_;
    std::vector<int> input_shape_cache_;
    dl::CudnnTensorDescriptor tensor_desc_;
    bool descriptor_configured_ { false };
};
