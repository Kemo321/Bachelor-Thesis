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
        const std::string message = std::string("cuDNN error at ") + file + ":" + std::to_string(line) + ": "
            + cudnnGetErrorString(status);
        log_error_message(message);
        throw std::runtime_error(message);
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

inline auto bind_cudnn_stream(cudaStream_t stream) -> void
{
    CHECK_CUDNN(cudnnSetStream(get_cudnn_handle(), stream));
}

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
    auto set_nchw(int n, int c, int h, int w, cudnnDataType_t data_type = CUDNN_DATA_FLOAT) -> void;

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
    auto set_nchw(int out_channels, int in_channels, int kernel_h, int kernel_w,
        cudnnDataType_t data_type = CUDNN_DATA_FLOAT) -> void;

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
    auto set_2d(int padding, int stride, cudnnDataType_t compute_type = CUDNN_DATA_FLOAT) -> void;
    auto set_math_type(cudnnMathType_t math_type) -> void;

private:
    cudnnConvolutionDescriptor_t desc_ { nullptr };
};

class CudnnActivationDescriptor
{
public:
    CudnnActivationDescriptor();
    ~CudnnActivationDescriptor();

    CudnnActivationDescriptor(const CudnnActivationDescriptor&) = delete;
    auto operator=(const CudnnActivationDescriptor&) -> CudnnActivationDescriptor& = delete;
    CudnnActivationDescriptor(CudnnActivationDescriptor&& other) noexcept;
    auto operator=(CudnnActivationDescriptor&& other) noexcept -> CudnnActivationDescriptor&;

    auto get() const -> cudnnActivationDescriptor_t;
    auto set(cudnnActivationMode_t mode, cudnnNanPropagation_t nan_opt, double coef) -> void;

private:
    cudnnActivationDescriptor_t desc_ { nullptr };
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

[[nodiscard]] inline auto cudnn_data_type(dl::Dtype dtype) -> cudnnDataType_t
{
    return dtype == dl::Dtype::Float16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
}

/**
 * 2D convolution via cuDNN (cross-correlation).
 *
 * Weights are NCHW filters [C_out, C_in, K, K]. Biases are [1, C_out, 1, 1]
 * for cudnnAddTensor.
 */
class Conv2d : public Layer
{
public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val,
        float inertia_val = 0.0F);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    void clip_gradients(float abs_bound, cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

private:
    auto configure_io_descriptors(int batch, int height, int width) -> void;
    auto select_algorithms() -> void;
    auto ensure_workspace(size_t bytes) -> void;

    dl::Tensor weights_;
    dl::Tensor biases_;
    std::optional<dl::Tensor> input_cache_;
    bool input_cache_ready_ { false };
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
    dl::CudnnActivationDescriptor activation_desc_;
    dl::CudaWorkspace workspace_;

    cudnnConvolutionFwdAlgo_t fwd_algo_ { CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM };
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_ { CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 };
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_ { CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 };
    bool algorithms_selected_ { false };
};
