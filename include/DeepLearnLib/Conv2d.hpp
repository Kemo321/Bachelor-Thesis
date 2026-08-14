#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cudnn.h>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace dl
{

inline auto check_cudnn(cudnnStatus_t status, const char* file, int line) -> void
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string("cuDNN error at ") + file + ":" + std::to_string(line) + ": " + cudnnGetErrorString(status));
    }
}

#ifndef CHECK_CUDNN
#define CHECK_CUDNN(call) ::dl::check_cudnn((call), __FILE__, __LINE__)
#endif

class CudnnContext
{
public:
    static auto handle() -> cudnnHandle_t;

    CudnnContext(const CudnnContext&) = delete;
    auto operator=(const CudnnContext&) -> CudnnContext& = delete;
    CudnnContext(CudnnContext&&) = delete;
    auto operator=(CudnnContext&&) -> CudnnContext& = delete;

private:
    CudnnContext();
    ~CudnnContext();

    cudnnHandle_t handle_ { nullptr };
};

auto get_cudnn_handle() -> cudnnHandle_t;

class CudnnTensorDescriptor
{
public:
    CudnnTensorDescriptor();
    ~CudnnTensorDescriptor();

    CudnnTensorDescriptor(const CudnnTensorDescriptor&) = delete;
    auto operator=(const CudnnTensorDescriptor&) -> CudnnTensorDescriptor& = delete;
    CudnnTensorDescriptor(CudnnTensorDescriptor&& other) noexcept;
    auto operator=(CudnnTensorDescriptor&& other) noexcept -> CudnnTensorDescriptor&;

    auto get() const -> cudnnTensorDescriptor_t;
    auto set_nchw(int n, int c, int h, int w) -> void;

private:
    cudnnTensorDescriptor_t desc_ { nullptr };
};

class CudnnFilterDescriptor
{
public:
    CudnnFilterDescriptor();
    ~CudnnFilterDescriptor();

    CudnnFilterDescriptor(const CudnnFilterDescriptor&) = delete;
    auto operator=(const CudnnFilterDescriptor&) -> CudnnFilterDescriptor& = delete;
    CudnnFilterDescriptor(CudnnFilterDescriptor&& other) noexcept;
    auto operator=(CudnnFilterDescriptor&& other) noexcept -> CudnnFilterDescriptor&;

    auto get() const -> cudnnFilterDescriptor_t;
    auto set_nchw(int out_channels, int in_channels, int kernel_h, int kernel_w) -> void;

private:
    cudnnFilterDescriptor_t desc_ { nullptr };
};

class CudnnConvolutionDescriptor
{
public:
    CudnnConvolutionDescriptor();
    ~CudnnConvolutionDescriptor();

    CudnnConvolutionDescriptor(const CudnnConvolutionDescriptor&) = delete;
    auto operator=(const CudnnConvolutionDescriptor&) -> CudnnConvolutionDescriptor& = delete;
    CudnnConvolutionDescriptor(CudnnConvolutionDescriptor&& other) noexcept;
    auto operator=(CudnnConvolutionDescriptor&& other) noexcept -> CudnnConvolutionDescriptor&;

    auto get() const -> cudnnConvolutionDescriptor_t;
    auto set_2d(int padding, int stride) -> void;

private:
    cudnnConvolutionDescriptor_t desc_ { nullptr };
};

class CudaWorkspace
{
public:
    auto ensure(size_t bytes) -> void;
    auto get() const -> void*;
    auto size() const -> size_t;

private:
    struct Deleter
    {
        void operator()(void* pointer) const;
    };

    std::unique_ptr<void, Deleter> ptr_;
    size_t bytes_ { 0 };
};

} // namespace dl

/**
 * @brief Conv2d layer implementing 2D convolution with cuDNN.
 *
 * Weights are stored as NCHW filters [C_out, C_in, K, K]. Biases are stored as
 * a broadcastable NCHW tensor [1, C_out, 1, 1] for cudnnAddTensor.
 */
class Conv2d : public Layer
{
public:
    /**
     * @brief Construct a Conv2d layer.
     *
     * @param in_channels Number of input channels (C_in).
     * @param out_channels Number of output channels / filters (C_out).
     * @param kernel_size Size of the square convolution kernel (K).
     * @param stride_val Stride for the convolution operation.
     * @param padding_val Padding applied to input on both sides.
     * @param inertia_val Momentum/inertia factor used in parameter updates.
     */
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val,
        float inertia_val = 0.0F);

    /**
     * @brief Forward pass of the convolutional layer.
     *
     * @param input_tensor Input tensor with shape [Batch, Channels_in, Height_in, Width_in].
     * @return Output tensor after convolution with shape [Batch, Channels_out, Height_out, Width_out].
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;

    /**
     * @brief Backward pass computing gradients w.r.t. inputs and parameters.
     *
     * @param output_error_derivative Gradient of the loss w.r.t. this layer's output.
     *        Expected shape: [Batch, Channels_out, Height_out, Width_out].
     * @return Gradient of the loss w.r.t. this layer's input. Shape: [Batch, Channels_in, Height_in, Width_in].
     */
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;

    /**
     * @brief Update parameters (weights and biases) using accumulated gradients.
     */
    void step() override;

    /**
     * @brief Retrieve current learnable parameters.
     *
     * @return Map of parameter name to tensor. Keys: "weights", "bias".
     */
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;

    /**
     * @brief Replace current parameters from an external source.
     *
     * @param params Map containing tensors for parameters. Expected keys: "weights", "bias".
     */
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;

    /**
     * @brief Conv2d parameters already reside on the GPU; CPU placement is rejected.
     */
    auto to(dl::Device device) -> void override;

private:
    auto configure_io_descriptors(int batch, int height, int width) -> void;
    auto select_algorithms() -> void;
    auto ensure_workspace(size_t bytes) -> void;

    dl::Tensor weights_;
    dl::Tensor biases_;
    std::optional<dl::Tensor> input_cache_;
    dl::Tensor weights_gradient_;
    dl::Tensor biases_gradient_;

    std::vector<int> input_shape_cache_;
    std::vector<int> output_shape_cache_;

    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    float inertia_;

    dl::CudnnTensorDescriptor input_desc_;
    dl::CudnnTensorDescriptor output_desc_;
    dl::CudnnTensorDescriptor bias_desc_;
    dl::CudnnFilterDescriptor filter_desc_;
    dl::CudnnConvolutionDescriptor conv_desc_;
    dl::CudaWorkspace workspace_;

    cudnnConvolutionFwdAlgo_t fwd_algo_ { CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM };
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_ { CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 };
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_ { CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 };
    bool algorithms_selected_ { false };
};
