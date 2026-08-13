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
    cudnnPoolingDescriptor_t desc_{ nullptr };
};

} // namespace dl

/**
 * @brief 2D max pooling layer implemented with cuDNN.
 *
 * Both the input and the pooling output are cached during the forward pass
 * because cudnnPoolingBackward requires x and y in addition to dy.
 */
class MaxPool2d : public Layer
{
public:
    /**
     * @brief Constructs a MaxPool2d layer.
     *
     * @param kernel_size_val Size of the square pooling window.
     * @param stride_val Stride of the pooling operation.
     */
    MaxPool2d(int kernel_size_val, int stride_val);

    /**
     * @brief Performs the forward pass of the max pooling layer.
     *
     * @param input_tensor Input tensor with shape [Batch, Channels, Height, Width].
     * @return Tensor with shape [Batch, Channels, PooledHeight, PooledWidth].
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;

    /**
     * @brief Performs the backward pass of the max pooling layer.
     *
     * @param output_error_derivative Gradient of the loss with respect to the output.
     * @return Gradient of the loss with respect to the input.
     */
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;

private:
    auto configure_descriptors(int batch, int channels, int height, int width) -> void;

    int kernel_size_;
    int stride_;

    std::optional<dl::Tensor> input_cache_;
    std::optional<dl::Tensor> output_cache_;
    std::vector<int> input_shape_cache_;
    std::vector<int> output_shape_cache_;

    dl::CudnnPoolingDescriptor pooling_desc_;
    dl::CudnnTensorDescriptor input_desc_;
    dl::CudnnTensorDescriptor output_desc_;
    bool descriptors_configured_{ false };
};
