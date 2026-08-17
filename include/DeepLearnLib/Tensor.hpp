#pragma once

#ifndef DEEPLEARNLIB_ENABLE_CUDA
#define DEEPLEARNLIB_ENABLE_CUDA 1
#endif

#include <algorithm>
#include <cstddef>
#if DEEPLEARNLIB_ENABLE_CUDA
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "DeepLearnLib/Precision.hpp"

namespace dl
{

auto log_error_message(const std::string& message) -> void;
auto log_info_message(const std::string& message) -> void;

enum class Device
{
    CPU,
    GPU
};

struct CpuDeleter
{
    void operator()(float* ptr) const
    {
        if (ptr)
        {
            ::operator delete(static_cast<void*>(ptr));
        }
    }
};

#if DEEPLEARNLIB_ENABLE_CUDA
struct CudaDeleter
{
    void operator()(float* ptr) const
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
};

inline auto check_cuda(cudaError_t status, const char* file, int line) -> void
{
    if (status != cudaSuccess)
    {
        const std::string message = std::string("CUDA error at ") + file + ":" + std::to_string(line) + ": "
            + cudaGetErrorString(status);
        log_error_message(message);
        throw std::runtime_error(message);
    }
}

inline auto check_cublas(cublasStatus_t status, const char* file, int line) -> void
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        const std::string message = std::string("cuBLAS error at ") + file + ":" + std::to_string(line) + ": "
            + cublasGetStatusString(status);
        log_error_message(message);
        throw std::runtime_error(message);
    }
}

#define CHECK_CUDA(call) ::dl::check_cuda((call), __FILE__, __LINE__)
#define CHECK_CUBLAS(call) ::dl::check_cublas((call), __FILE__, __LINE__)

class CublasContext
{
public:
    static auto handle() -> cublasHandle_t;

    CublasContext(const CublasContext&) = delete;
    auto operator=(const CublasContext&) -> CublasContext& = delete;
    CublasContext(CublasContext&&) = delete;
    auto operator=(CublasContext&&) -> CublasContext& = delete;

private:
    CublasContext();
    ~CublasContext();

    cublasHandle_t handle_ { nullptr };
};

auto get_cublas_handle() -> cublasHandle_t;

[[nodiscard]] auto current_stream() -> cudaStream_t;
auto set_current_stream(cudaStream_t stream) -> void;

class StreamGuard
{
public:
    explicit StreamGuard(cudaStream_t stream);
    ~StreamGuard();

    StreamGuard(const StreamGuard&) = delete;
    auto operator=(const StreamGuard&) -> StreamGuard& = delete;
    StreamGuard(StreamGuard&&) = delete;
    auto operator=(StreamGuard&&) -> StreamGuard& = delete;

private:
    cudaStream_t previous_ { 0 };
};

struct PinnedHostDeleter
{
    cudaStream_t stream { 0 };

    void operator()(float* ptr) const
    {
        if (ptr == nullptr)
        {
            return;
        }
        static_cast<void>(cudaStreamSynchronize(stream));
        static_cast<void>(cudaFreeHost(ptr));
    }
};

class UniqueCudaStream
{
public:
    UniqueCudaStream()
    {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~UniqueCudaStream()
    {
        if (stream_ != nullptr)
        {
            static_cast<void>(cudaStreamSynchronize(stream_));
            static_cast<void>(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    UniqueCudaStream(const UniqueCudaStream&) = delete;
    auto operator=(const UniqueCudaStream&) -> UniqueCudaStream& = delete;
    UniqueCudaStream(UniqueCudaStream&&) = delete;
    auto operator=(UniqueCudaStream&&) -> UniqueCudaStream& = delete;

    [[nodiscard]] auto get() const -> cudaStream_t
    {
        return stream_;
    }

private:
    cudaStream_t stream_ { nullptr };
};

inline auto memcpy_d2d_on_current(void* dst, const void* src, size_t bytes) -> void
{
    if (bytes == 0)
    {
        return;
    }
    CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, current_stream()));
}
#endif

/**
 * Dense tensor with CUDA-managed storage (cudaMalloc via CudaDeleter).
 *
 * Default dtype is FP32. Mixed-precision training can allocate FP16 (`__half`)
 * storage so Conv2d/FullyConnected can run on Tensor Cores.
 *
 * Elementwise ops and GEMM stay on the device; host copies happen only through
 * to_host / from_host (which always speak IEEE-754 float on the host).
 */
class Tensor
{
public:
    Tensor();
    explicit Tensor(std::vector<int> shape, Device device = Device::CPU, Dtype dtype = Dtype::Float32);

    Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data, Device device,
        Dtype dtype = Dtype::Float32);

    ~Tensor() = default;

    Tensor(const Tensor&) = delete;
    auto operator=(const Tensor&) -> Tensor& = delete;

    Tensor(Tensor&&) noexcept = default;
    auto operator=(Tensor&&) noexcept -> Tensor& = default;

    // clang-format off
    // cppcheck-suppress unusedFunction
    auto get_shape() const -> const std::vector<int>&;
    auto get_strides() const -> const std::vector<int>&;
    auto get_size() const -> size_t;
    auto get_device() const -> Device;
    auto get_dtype() const -> Dtype;
    auto element_size() const -> std::size_t;
    auto nbytes() const -> std::size_t;
    auto get_data() const -> const float*;
    auto data() -> float*;
    auto data() const -> const float*;
#if DEEPLEARNLIB_ENABLE_CUDA
    auto half_data() -> __half*;
    auto half_data() const -> const __half*;
    auto to_dtype(Dtype dtype, cudaStream_t stream = 0) const -> Tensor;
#endif
    // clang-format on

    auto matmul(const Tensor& other) const -> Tensor;

    auto operator+(const Tensor& other) const -> Tensor;
    auto operator-(const Tensor& other) const -> Tensor;
    auto operator*(const Tensor& other) const -> Tensor;

    auto operator*(float scalar) const -> Tensor;
    auto operator+(float scalar) const -> Tensor;

    auto clamp(float lo, float hi) const -> Tensor;

    [[nodiscard]] auto has_non_finite() const -> bool;
    auto assert_finite(const char* context) const -> void;

    auto sum(int dim = -1) const -> Tensor;

    auto view(const std::vector<int>& new_shape) const -> Tensor;
    auto transpose() const -> Tensor;

    static auto zeros_like(const Tensor& other) -> Tensor;

    [[nodiscard]] auto describe() const -> std::string;

#if DEEPLEARNLIB_ENABLE_CUDA
    auto to_host(cudaStream_t stream = 0) const -> std::vector<float>;
    static auto from_host(const std::vector<int>& shape, const std::vector<float>& host_data,
        Device device = Device::GPU, cudaStream_t stream = 0, Dtype dtype = Dtype::Float32) -> Tensor;
    static auto from_host(const std::vector<int>& shape, const float* host_data, Device device = Device::GPU,
        cudaStream_t stream = 0, Dtype dtype = Dtype::Float32) -> Tensor;
#else
    auto to_host() const -> std::vector<float>;
    static auto from_host(const std::vector<int>& shape, const std::vector<float>& host_data,
        Device device = Device::GPU, Dtype dtype = Dtype::Float32) -> Tensor;
    static auto from_host(const std::vector<int>& shape, const float* host_data, Device device = Device::GPU,
        Dtype dtype = Dtype::Float32) -> Tensor;
#endif

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    size_t size_ = 0;
    Device device_;
    Dtype dtype_ { Dtype::Float32 };

    std::shared_ptr<float> data_;
#if DEEPLEARNLIB_ENABLE_CUDA
    std::unique_ptr<float, PinnedHostDeleter> h2d_staging_;
#endif

    auto compute_strides() -> void;
    auto is_contiguous() const -> bool;
    auto ensure_gpu(const char* op_name) const -> void;
    auto ensure_binary_op(const Tensor& other, const char* op_name) const -> void;
};

inline auto format_shape(const std::vector<int>& shape) -> std::string
{
    std::string text = "[";
    for (std::size_t index = 0; index < shape.size(); ++index)
    {
        if (index > 0)
        {
            text += ", ";
        }
        text += std::to_string(shape[index]);
    }
    text += "]";
    return text;
}

} // namespace dl
