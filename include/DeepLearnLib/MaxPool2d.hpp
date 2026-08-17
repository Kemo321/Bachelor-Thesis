#pragma once

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cudnn.h>
#include <optional>
#include <vector>

namespace dl
{

class CudnnPoolingDescriptor
{
public:
    CudnnPoolingDescriptor();
    ~CudnnPoolingDescriptor();

    CudnnPoolingDescriptor(const CudnnPoolingDescriptor&) = delete;
    auto operator=(const CudnnPoolingDescriptor&) -> CudnnPoolingDescriptor& = delete;
    CudnnPoolingDescriptor(CudnnPoolingDescriptor&& other) noexcept;
    auto operator=(CudnnPoolingDescriptor&& other) noexcept -> CudnnPoolingDescriptor&;

    auto get() const -> cudnnPoolingDescriptor_t;
    auto set_max_2d(int window, int stride, int padding = 0) -> void;

private:
    cudnnPoolingDescriptor_t desc_ { nullptr };
};

} // namespace dl

/**
 * 2D max pooling via cuDNN.
 *
 * Forward caches x and y because cudnnPoolingBackward requires both in addition to dy.
 */
class MaxPool2d : public Layer
{
public:
    MaxPool2d(int kernel_size_val, int stride_val);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;

private:
    auto configure_descriptors(int batch, int channels, int height, int width, dl::Dtype dtype) -> void;

    int kernel_size_;
    int stride_;

    std::optional<dl::Tensor> input_cache_;
    std::optional<dl::Tensor> output_cache_;
    std::optional<dl::Tensor> grad_input_cache_;
    bool caches_ready_ { false };
    std::vector<int> input_shape_cache_;
    std::vector<int> output_shape_cache_;

    dl::CudnnPoolingDescriptor pooling_desc_;
    dl::CudnnTensorDescriptor input_desc_;
    dl::CudnnTensorDescriptor output_desc_;
    bool descriptors_configured_ { false };
};
