#pragma once

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <optional>
#include <vector>

/**
 * Softmax layer via cuDNN (`CUDNN_SOFTMAX_ACCURATE`, `MODE_CHANNEL`).
 *
 * `CUDNN_SOFTMAX_ACCURATE` applies the stable max-subtraction form
 * `exp(x - max(x)) / sum(exp(x - max(x)))` to avoid exponential overflow.
 */
class Softmax : public Layer
{
public:
    Softmax() = default;

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;

private:
    auto configure_descriptor(const dl::Tensor& tensor) -> void;

    std::optional<dl::Tensor> output_cache_;
    std::vector<int> input_shape_cache_;
    dl::CudnnTensorDescriptor tensor_desc_;
    bool descriptor_configured_ { false };
};
