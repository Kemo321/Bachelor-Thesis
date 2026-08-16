#pragma once

#ifndef DEEPLEARNLIB_ENABLE_CUDA
#define DEEPLEARNLIB_ENABLE_CUDA 1
#endif

#include <algorithm>
#include <cstddef>
#if DEEPLEARNLIB_ENABLE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
            delete[] ptr;
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
#endif

/**
 * Dense float tensor with CUDA-managed storage (cudaMalloc via CudaDeleter).
 *
 * Elementwise ops and GEMM stay on the device; host copies happen only through
 * to_host / from_host.
 */
class Tensor
{
public:
    explicit Tensor(std::vector<int> shape, Device device = Device::CPU);

    Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data, Device device);

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
    auto get_data() const -> const float*;
    auto data() -> float*;
    auto data() const -> const float*;
    // clang-format on

    auto matmul(const Tensor& other) const -> Tensor;

    auto operator+(const Tensor& other) const -> Tensor;
    auto operator-(const Tensor& other) const -> Tensor;
    auto operator*(const Tensor& other) const -> Tensor;

    auto operator*(float scalar) const -> Tensor;
    auto operator+(float scalar) const -> Tensor;

    auto clamp(float lo, float hi) const -> Tensor;

    auto sum(int dim = -1) const -> Tensor;

    auto view(const std::vector<int>& new_shape) const -> Tensor;
    auto transpose() const -> Tensor;

    static auto zeros_like(const Tensor& other) -> Tensor;

    auto to_host() const -> std::vector<float>;
    static auto from_host(const std::vector<int>& shape, const std::vector<float>& host_data,
        Device device = Device::GPU) -> Tensor;
    static auto from_host(const std::vector<int>& shape, const float* host_data, Device device = Device::GPU)
        -> Tensor;

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    size_t size_ = 0;
    Device device_;

    std::shared_ptr<float> data_;

    auto compute_strides() -> void;
    auto is_contiguous() const -> bool;
    auto ensure_gpu(const char* op_name) const -> void;
    auto ensure_binary_op(const Tensor& other, const char* op_name) const -> void;
};

} // namespace dl
