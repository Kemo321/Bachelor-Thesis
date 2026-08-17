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
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "DeepLearnLib/Precision.hpp"

namespace dl
{

auto log_error_message(const std::string& message) -> void;
auto log_info_message(const std::string& message) -> void;
auto log_debug_message(const std::string& message) -> void;
auto log_flush() -> void;

/** Host vs GPU placement for `dl::Tensor` storage. Training uses `GPU`. */
enum class Device
{
    CPU,
    GPU
};

/** Host `operator delete` for CPU tensors. */
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
/** `cudaFree` deleter for GPU tensors allocated with `cudaMalloc`. */
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

/**
 * @brief Process-wide cuBLAS handle plus a persistent workspace allocation.
 *
 * The workspace avoids per-GEMM `cublasSetWorkspace` allocations on the
 * training hot path.
 */
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
    void* workspace_ { nullptr };
};

auto get_cublas_handle() -> cublasHandle_t;

[[nodiscard]] auto current_stream() -> cudaStream_t;
auto set_current_stream(cudaStream_t stream) -> void;

/**
 * @brief RAII switch of `dl::current_stream()` for a lexical scope.
 *
 * Restores the previous stream in the destructor so nested layers can bind
 * cuDNN/cuBLAS without leaking stream state.
 */
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

/**
 * @brief Non-blocking CUDA stream with destroy-on-scope-exit.
 *
 * Used by dataloaders and double-buffered training loops to overlap H2D with
 * compute. The destructor synchronises then destroys the stream.
 */
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
 * @brief Dense tensor with CUDA-managed storage (`cudaMalloc` via `CudaDeleter`).
 *
 * Default dtype is FP32. Mixed-precision training can allocate FP16 (`__half`)
 * storage so Conv2d/FullyConnected can run on Tensor Cores.
 *
 * Elementwise ops and GEMM stay on the device. Host copies happen only through
 * `to_host` / `from_host` (IEEE-754 float on the host). Layers cache activations
 * and workspaces with `ensure()` so a stable batch size trades VRAM for fewer
 * allocator stalls — the same time/memory tradeoff LibTorch uses internally.
 */
class Tensor
{
public:
    /** Empty tensor (CPU, no storage). */
    Tensor();
    /**
     * @brief Allocate a dense tensor.
     * @param shape Dimension sizes (row-major).
     * @param device CPU or GPU. Training tensors should be GPU.
     * @param dtype Storage type (`Float32` or `Float16`).
     */
    explicit Tensor(std::vector<int> shape, Device device = Device::CPU, Dtype dtype = Dtype::Float32);

    /**
     * @brief Wrap existing storage (used for views). Does not copy.
     * @param shape Logical shape.
     * @param strides Byte/element strides matching @p data.
     * @param data Shared storage (typically the parent tensor's buffer).
     * @param device Device of @p data.
     * @param dtype Element type of @p data.
     */
    Tensor(std::vector<int> shape, std::vector<int> strides, std::shared_ptr<float> data, Device device,
        Dtype dtype = Dtype::Float32);

    ~Tensor() = default;

    Tensor(const Tensor&) = delete;
    auto operator=(const Tensor&) -> Tensor& = delete;

    Tensor(Tensor&&) noexcept = default;
    auto operator=(Tensor&&) noexcept -> Tensor& = default;

    // clang-format off
    // cppcheck-suppress unusedFunction
    /** @return Logical shape. */
    auto get_shape() const -> const std::vector<int>&;
    /** @return Row-major strides in elements. */
    auto get_strides() const -> const std::vector<int>&;
    /** @return Number of elements. */
    auto get_size() const -> size_t;
    auto get_device() const -> Device;
    auto get_dtype() const -> Dtype;
    /** @return Bytes per element. */
    auto element_size() const -> std::size_t;
    /** @return `get_size() * element_size()`. */
    auto nbytes() const -> std::size_t;
    auto get_data() const -> const float*;
    /**
     * @brief Mutable device/host pointer (FP32 view).
     * @note In-place kernels write through this pointer to avoid new allocations.
     */
    auto data() -> float*;
    auto data() const -> const float*;
#if DEEPLEARNLIB_ENABLE_CUDA
    auto half_data() -> __half*;
    auto half_data() const -> const __half*;
    /**
     * @brief Convert storage dtype, allocating a new tensor.
     * @param dtype Destination type.
     * @param stream CUDA stream.
     */
    auto to_dtype(Dtype dtype, cudaStream_t stream = 0) const -> Tensor;
#endif
    // clang-format on

    /**
     * @brief Row-major GEMM: C = op(A) * op(B). Allocates a new result.
     * @param other Right-hand operand B.
     * @param transpose_a If true, use A^T without a physical transpose (cuBLAS `OP_T`).
     * @param transpose_b If true, use B^T without a physical transpose.
     * @return Newly allocated product C.
     */
    auto matmul(const Tensor& other, bool transpose_a = false, bool transpose_b = false) const -> Tensor;
    /**
     * @brief C = op(A) * op(B) + beta * C into an existing buffer.
     * @param other Right-hand operand B.
     * @param out Pre-allocated result with the GEMM shape and matching dtype.
     * @param transpose_a Logical transpose of A.
     * @param transpose_b Logical transpose of B.
     * @param beta Scale for the existing contents of @p out (`0` overwrites).
     * @return Reference to @p out.
     *
     * @note No allocation. This is the hot path for FullyConnected forward/backward.
     */
    auto matmul_into(const Tensor& other, Tensor& out, bool transpose_a = false, bool transpose_b = false,
        float beta = 0.0F) const -> Tensor&;

    /**
     * @brief Grow @p slot only when shape, device, or dtype change.
     * @param slot Optional cache owned by a layer.
     * @param shape Desired shape.
     * @param device Desired device.
     * @param dtype Desired storage type.
     * @return Reference to the (possibly newly allocated) tensor in @p slot.
     *
     * @note Avoids `cudaMalloc` in the training loop when the batch size is stable.
     *       Higher VRAM (static caches) is intentional: it matches LibTorch speed.
     */
    static auto ensure(std::optional<Tensor>& slot, const std::vector<int>& shape, Device device,
        Dtype dtype = Dtype::Float32) -> Tensor&;

    auto operator+(const Tensor& other) const -> Tensor;
    auto operator-(const Tensor& other) const -> Tensor;
    auto operator*(const Tensor& other) const -> Tensor;

    auto operator*(float scalar) const -> Tensor;
    auto operator+(float scalar) const -> Tensor;

    /**
     * @brief In-place: `this += other`.
     * @note No allocation.
     */
    auto add_(const Tensor& other) -> Tensor&;
    /**
     * @brief In-place: `this *= scalar`.
     * @note No allocation.
     */
    auto mul_(float scalar) -> Tensor&;
    /**
     * @brief `out = *this * other` (elementwise).
     * @param other Broadcast-compatible tensor.
     * @param out Pre-allocated destination with matching shape and dtype.
     * @note `out` must already match shape and dtype; no allocation.
     */
    auto mul_into(const Tensor& other, Tensor& out) const -> Tensor&;
    /**
     * @brief In-place: `this += scale * other`.
     * @note No allocation.
     */
    auto add_scaled_(const Tensor& other, float scale) -> Tensor&;
    /**
     * @brief Fused SGD: `this -= lr * clip(grad + decay * this, [-clip, clip])`.
     * @param grad Parameter gradient (same shape).
     * @param lr Learning rate.
     * @param decay L2 weight decay coefficient.
     * @param clip Absolute clip; `<= 0` disables clipping.
     * @return `*this`.
     *
     * @note No extra buffers. Weight decay is applied here, not in `backward()`.
     */
    auto sgd_update_(const Tensor& grad, float lr, float decay, float clip = 0.0F) -> Tensor&;
    /**
     * @brief In-place: each row of `[B, C] +=` bias of shape `[1, C]` or `[C]`.
     * @note No allocation.
     */
    auto add_row_(const Tensor& bias) -> Tensor&;
    /**
     * @brief In-place reduction: `this[j] = beta * this[j] + sum_i matrix[i, j]`.
     * @param matrix Rank-2 tensor `[B, C]`.
     * @param beta Scale for existing contents of `this` (`[1, C]` or `[C]`).
     * @note No allocation of a reduction output.
     */
    auto add_sum_rows_(const Tensor& matrix, float beta = 0.0F) -> Tensor&;

    auto clamp(float lo, float hi) const -> Tensor;
    /**
     * @brief In-place clamp to `[lo, hi]`.
     * @note No allocation.
     */
    auto clamp_(float lo, float hi) -> Tensor&;

    [[nodiscard]] auto has_non_finite() const -> bool;
    auto assert_finite(const char* context) const -> void;

    auto sum(int dim = -1) const -> Tensor;

    /**
     * @brief View with a new shape (shared storage when contiguous).
     * @param new_shape Dimensions whose product equals `get_size()`.
     */
    auto view(const std::vector<int>& new_shape) const -> Tensor;
    /**
     * @brief Cheap alias of this tensor (same storage, shape, strides).
     * @note Avoids a device-to-device copy of layer inputs.
     */
    [[nodiscard]] auto as_view() const -> Tensor;
    auto transpose() const -> Tensor;

    static auto zeros_like(const Tensor& other) -> Tensor;

    [[nodiscard]] auto describe() const -> std::string;

#if DEEPLEARNLIB_ENABLE_CUDA
    /**
     * @brief Copy device storage to a host `std::vector<float>` (FP32).
     * @param stream CUDA stream; the copy is synchronised before return.
     * @note Host sync — use only for logging, metrics, or checkpoints.
     */
    auto to_host(cudaStream_t stream = 0) const -> std::vector<float>;
    /**
     * @brief Upload host FP32 data to a new device (or CPU) tensor.
     * @param shape Logical shape.
     * @param host_data Contiguous row-major values.
     * @param device Destination.
     * @param stream CUDA stream for H2D.
     * @param dtype Device storage type (may convert to FP16).
     */
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
