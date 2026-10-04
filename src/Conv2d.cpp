#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

constexpr int kFillThreads = 256;

struct UniformFill
{
    float low;
    float high;
    unsigned long long seed;

    // Maps an index onto a value in [low, high) by hashing. Weight initialisation
    // does not call a host generator and does not synchronise the device.
    __host__ __device__ auto operator()(int index) const -> float
    {
        unsigned long long hash = seed + (static_cast<unsigned long long>(index) + 1ULL) * 0x9E3779B97F4A7C15ULL;
        hash ^= hash >> 30U;
        hash *= 0xBF58476D1CE4E5B9ULL;
        hash ^= hash >> 27U;
        hash *= 0x94D049BB133111EBULL;
        hash ^= hash >> 31U;
        const float unit = static_cast<float>(hash & 0xFFFFFFULL) / static_cast<float>(0x1000000ULL);
        return low + ((high - low) * unit);
    }
};

// Applies UniformFill across the tensor elements. The range and seed travel in the
// functor, so the kernel does not read extra buffers.
__global__ void uniform_fill_kernel(float* out, int count, UniformFill fill)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
    {
        out[index] = fill(index);
    }
}

// Launches the uniform-initialisation kernel. An empty tensor is skipped so the
// launch does not use a grid of zero blocks.
auto fill_uniform(dl::Tensor& tensor, float low, float high, unsigned long long seed) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    const int count = static_cast<int>(tensor.get_size());
    const dim3 grid(static_cast<unsigned int>((count + kFillThreads - 1) / kFillThreads));
    uniform_fill_kernel<<<grid, kFillThreads, 0, dl::current_stream()>>>(
        tensor.data(), count, UniformFill { low, high, seed });
    CHECK_CUDA(cudaGetLastError());
}

// Zeros the buffer on the current stream. Gradients start at zero and do not
// need a separate fill kernel.
auto fill_zero(dl::Tensor& tensor) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    CHECK_CUDA(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), dl::current_stream()));
    CHECK_CUDA(cudaGetLastError());
}

// Creates or resizes the velocity buffer and zeros it when the shape changes.
// SGD momentum needs a velocity with the weight's shape; a fresh allocation left
// unzeroed would be garbage.
auto ensure_zero_like(std::optional<dl::Tensor>& slot, const dl::Tensor& like) -> dl::Tensor&
{
    if (!slot.has_value() || slot->get_shape() != like.get_shape() || slot->get_dtype() != like.get_dtype())
    {
        slot = dl::Tensor(like.get_shape(), like.get_device(), like.get_dtype());
        fill_zero(*slot);
    }
    return *slot;
}

// Rejects a tensor that is off the GPU or not rank 4. The cuDNN convolution reads NCHW layout from the descriptor.
auto require_gpu_nchw(const dl::Tensor& tensor, const char* name) -> void
{
    if (tensor.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " must reside on the GPU");
    }
    if (tensor.get_shape().size() != 4)
    {
        throw std::runtime_error(std::string(name) + " must have NCHW rank 4");
    }
    if (tensor.get_size() > 0 && tensor.data() == nullptr)
    {
        throw std::runtime_error(std::string(name) + " has a null device pointer");
    }
}

// Copies weights device-to-device, converting on the current stream when the
// dtype differs. The store must not change the filter shape.
auto copy_same_size(dl::Tensor& dst, const dl::Tensor& src, const char* name) -> void
{
    if (src.get_device() != dl::Device::GPU || dst.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " requires GPU tensors");
    }
    if (src.get_size() != dst.get_size())
    {
        throw std::runtime_error(std::string(name) + " tensor size mismatch");
    }
    if (src.get_size() == 0)
    {
        return;
    }
    if (src.get_dtype() == dst.get_dtype())
    {
        dl::memcpy_d2d_on_current(dst.data(), src.data(), src.nbytes());
        return;
    }
    const dl::Tensor converted = src.to_dtype(dst.get_dtype(), dl::current_stream());
    dl::memcpy_d2d_on_current(dst.data(), converted.data(), dst.nbytes());
}

constexpr int kMaxAlgoResults = 10;

// Returns the cuDNN workspace budget. A reserve is left so the chosen algorithm
// does not take all free device memory and fail the next allocation.
auto workspace_budget() -> size_t
{
    size_t free_bytes { 0 };
    size_t total_bytes { 0 };
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    constexpr size_t kReserve = 64ULL * 1024ULL * 1024ULL;
    // Keep 64 MiB free. Below that reserve, use half of the free memory.
    if (free_bytes > kReserve)
    {
        return free_bytes - kReserve;
    }
    return free_bytes / 2U;
}

// Takes the first cuDNN algorithm that succeeded and fits the budget.
// v7 returns results fastest-first, so the first fit is the selection.
template <typename PerfT>
auto pick_perf(const PerfT* perfs, int count, size_t budget) -> const PerfT*
{
    // v7 lists algorithms fastest-first. Keep the first success that fits the workspace budget.
    for (int index = 0; index < count; ++index)
    {
        if (perfs[index].status == CUDNN_STATUS_SUCCESS && perfs[index].memory <= budget)
        {
            return &perfs[index];
        }
    }
    return nullptr;
}

} // namespace

namespace dl
{

// Creates the process cuDNN handle. One context serves every layer, so
// cudnnCreate is not called on every convolution.
CudnnContext::CudnnContext()
{
    // The handle lives for the process. Layers share it instead of creating one per convolution.
    CHECK_CUDNN(cudnnCreate(&handle_));
}

// Destroys the cuDNN handle if construction succeeded. The destructor must not
// throw, so a cuDNN error here is deliberately ignored.
CudnnContext::~CudnnContext()
{
    if (handle_ != nullptr)
    {
        // Ignored on purpose: a destructor must not throw.
        static_cast<void>(cudnnDestroy(handle_));
        handle_ = nullptr;
    }
}

// Returns the handle from a function-local static instance. The first call
// creates the context; later calls only return the same handle.
auto CudnnContext::handle() -> cudnnHandle_t
{
    // First call builds the context; later calls return this same handle.
    static CudnnContext context;
    return context.handle_;
}

// Hands the shared cuDNN handle to the layers. A layer does not keep its own copy of the context.
auto get_cudnn_handle() -> cudnnHandle_t
{
    // Borrowed handle. The layer does not own a cuDNN context.
    return CudnnContext::handle();
}

// Creates a cuDNN tensor descriptor. The wrapper lives with the layer, so forward
// only replaces the dimensions, not the handle itself.
CudnnTensorDescriptor::CudnnTensorDescriptor()
{
    // Created with the owner. Forward rewrites NCHW on this handle; it does not create another.
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc_));
}

// Destroys the tensor descriptor if construction succeeded. The destructor must not
// throw, so a cuDNN error here is deliberately ignored.
CudnnTensorDescriptor::~CudnnTensorDescriptor()
{
    if (desc_ != nullptr)
    {
        // Ignored on purpose: a destructor must not throw.
        static_cast<void>(cudnnDestroyTensorDescriptor(desc_));
    }
}

// Takes ownership of the tensor descriptor and nulls the source. Otherwise the
// source destructor would free the same handle a second time.
CudnnTensorDescriptor::CudnnTensorDescriptor(CudnnTensorDescriptor&& other) noexcept
    : desc_(other.desc_)
{
    // The source must not destroy the handle this object now owns.
    other.desc_ = nullptr;
}

// Releases this tensor descriptor and takes the other one. The identity check
// guards against a double destroy on self-assignment.
auto CudnnTensorDescriptor::operator=(CudnnTensorDescriptor&& other) noexcept -> CudnnTensorDescriptor&
{
    // Self-assignment would destroy the handle and then use it.
    if (this != &other)
    {
        if (desc_ != nullptr)
        {
            static_cast<void>(cudnnDestroyTensorDescriptor(desc_));
        }
        desc_ = other.desc_;
        // The source must not destroy the handle this object now owns.
        other.desc_ = nullptr;
    }
    return *this;
}

// Returns the raw tensor-descriptor handle for cuDNN calls.
auto CudnnTensorDescriptor::get() const -> cudnnTensorDescriptor_t
{
    return desc_;
}

// Writes NCHW layout and the data type into the descriptor. Convolution and
// batch-norm read the dimensions from here, not from the tensor metadata.
auto CudnnTensorDescriptor::set_nchw(int n, int c, int h, int w, cudnnDataType_t data_type) -> void
{
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, data_type, n, c, h, w));
}

// Creates a cuDNN filter descriptor. The kernel shape is fixed, so the handle is created once.
CudnnFilterDescriptor::CudnnFilterDescriptor()
{
    // Created once with the layer. The kernel shape does not change between forwards.
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&desc_));
}

// Destroys the filter descriptor if construction succeeded. The destructor must not
// throw, so a cuDNN error here is deliberately ignored.
CudnnFilterDescriptor::~CudnnFilterDescriptor()
{
    if (desc_ != nullptr)
    {
        // Ignored on purpose: a destructor must not throw.
        static_cast<void>(cudnnDestroyFilterDescriptor(desc_));
    }
}

// Takes ownership of the filter descriptor and nulls the source. Otherwise the
// source destructor would free the same handle a second time.
CudnnFilterDescriptor::CudnnFilterDescriptor(CudnnFilterDescriptor&& other) noexcept
    : desc_(other.desc_)
{
    // The source must not destroy the handle this object now owns.
    other.desc_ = nullptr;
}

// Releases this filter descriptor and takes the other one. The identity check
// guards against a double destroy on self-assignment.
auto CudnnFilterDescriptor::operator=(CudnnFilterDescriptor&& other) noexcept -> CudnnFilterDescriptor&
{
    // Self-assignment would destroy the handle and then use it.
    if (this != &other)
    {
        if (desc_ != nullptr)
        {
            static_cast<void>(cudnnDestroyFilterDescriptor(desc_));
        }
        desc_ = other.desc_;
        // The source must not destroy the handle this object now owns.
        other.desc_ = nullptr;
    }
    return *this;
}

// Returns the raw filter-descriptor handle for cuDNN calls.
auto CudnnFilterDescriptor::get() const -> cudnnFilterDescriptor_t
{
    return desc_;
}

// Writes the NCHW kernel shape and the data type. Output channels are the K dimension of the cuDNN filter.
auto CudnnFilterDescriptor::set_nchw(int out_channels, int in_channels, int kernel_h, int kernel_w,
    cudnnDataType_t data_type) -> void
{
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(desc_, data_type, CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel_h,
        kernel_w));
}

// Creates a cuDNN convolution descriptor. Padding and stride are fixed for the
// layer, so the handle is not created on every forward.
CudnnConvolutionDescriptor::CudnnConvolutionDescriptor()
{
    // Padding and stride are fixed, so this handle outlives individual forwards.
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&desc_));
}

// Destroys the convolution descriptor if construction succeeded. The destructor must
// not throw, so a cuDNN error here is deliberately ignored.
CudnnConvolutionDescriptor::~CudnnConvolutionDescriptor()
{
    if (desc_ != nullptr)
    {
        // Ignored on purpose: a destructor must not throw.
        static_cast<void>(cudnnDestroyConvolutionDescriptor(desc_));
    }
}

// Takes ownership of the convolution descriptor and nulls the source. Otherwise the
// source destructor would free the same handle a second time.
CudnnConvolutionDescriptor::CudnnConvolutionDescriptor(CudnnConvolutionDescriptor&& other) noexcept
    : desc_(other.desc_)
{
    // The source must not destroy the handle this object now owns.
    other.desc_ = nullptr;
}

// Releases this convolution descriptor and takes the other one. The identity check
// guards against a double destroy on self-assignment.
auto CudnnConvolutionDescriptor::operator=(CudnnConvolutionDescriptor&& other) noexcept -> CudnnConvolutionDescriptor&
{
    // Self-assignment would destroy the handle and then use it.
    if (this != &other)
    {
        if (desc_ != nullptr)
        {
            static_cast<void>(cudnnDestroyConvolutionDescriptor(desc_));
        }
        desc_ = other.desc_;
        // The source must not destroy the handle this object now owns.
        other.desc_ = nullptr;
    }
    return *this;
}

// Returns the raw convolution-descriptor handle for cuDNN calls.
auto CudnnConvolutionDescriptor::get() const -> cudnnConvolutionDescriptor_t
{
    return desc_;
}

// Sets padding, stride, cross-correlation, and the accumulation type. Dilation
// stays 1 because the layer does not expose a kernel-dilation parameter.
auto CudnnConvolutionDescriptor::set_2d(int padding, int stride, cudnnDataType_t compute_type) -> void
{
    // Dilation stays 1. The layer has no kernel-dilation parameter. The mode is cross-correlation.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(desc_, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION,
        compute_type));
}

// Selects the convolution math mode, for example TF32. cuDNN picks an
// implementation for that mode, so it must be set before the algorithm search.
auto CudnnConvolutionDescriptor::set_math_type(cudnnMathType_t math_type) -> void
{
    // The algorithm search reads this mode, so it has to be set first.
    CHECK_CUDNN(cudnnSetConvolutionMathType(desc_, math_type));
}

// Creates a cuDNN activation descriptor. Convolution-with-bias needs one even
// when the activation is the identity.
CudnnActivationDescriptor::CudnnActivationDescriptor()
{
    // Lives with the layer. Convolution-with-bias requires it even for an identity activation.
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&desc_));
}

// Destroys the activation descriptor if construction succeeded. The destructor must
// not throw, so a cuDNN error here is deliberately ignored.
CudnnActivationDescriptor::~CudnnActivationDescriptor()
{
    if (desc_ != nullptr)
    {
        // Ignored on purpose: a destructor must not throw.
        static_cast<void>(cudnnDestroyActivationDescriptor(desc_));
    }
}

// Takes ownership of the activation descriptor and nulls the source. Otherwise the
// source destructor would free the same handle a second time.
CudnnActivationDescriptor::CudnnActivationDescriptor(CudnnActivationDescriptor&& other) noexcept
    : desc_(other.desc_)
{
    // The source must not destroy the handle this object now owns.
    other.desc_ = nullptr;
}

// Releases this activation descriptor and takes the other one. The identity check
// guards against a double destroy on self-assignment.
auto CudnnActivationDescriptor::operator=(CudnnActivationDescriptor&& other) noexcept -> CudnnActivationDescriptor&
{
    // Self-assignment would destroy the handle and then use it.
    if (this != &other)
    {
        if (desc_ != nullptr)
        {
            static_cast<void>(cudnnDestroyActivationDescriptor(desc_));
        }
        desc_ = other.desc_;
        // The source must not destroy the handle this object now owns.
        other.desc_ = nullptr;
    }
    return *this;
}

// Returns the raw activation-descriptor handle for cuDNN calls.
auto CudnnActivationDescriptor::get() const -> cudnnActivationDescriptor_t
{
    return desc_;
}

// Writes the activation mode, the NaN policy, and the coefficient. Conv2d sets
// the identity so convolution-with-bias stays linear.
auto CudnnActivationDescriptor::set(cudnnActivationMode_t mode, cudnnNanPropagation_t nan_opt, double coef) -> void
{
    CHECK_CUDNN(cudnnSetActivationDescriptor(desc_, mode, nan_opt, coef));
}

// Frees the cuDNN workspace buffer. The deleter sits in a unique_ptr so cudaFree
// runs when the workspace dies, rather than by hand on every ensure.
void CudaWorkspace::Deleter::operator()(void* pointer) const
{
    if (pointer != nullptr)
    {
        // Runs when the workspace is destroyed or replaced, not on every ensure.
        static_cast<void>(cudaFree(pointer));
    }
}

// Grows the workspace when cuDNN reports a larger size. A smaller request
// keeps the old allocation so memory is not freed on every forward.
auto CudaWorkspace::ensure(size_t bytes) -> void
{
    // A smaller request keeps the existing block.
    if (bytes <= bytes_)
    {
        return;
    }
    ptr_.reset();
    bytes_ = 0;
    if (bytes == 0)
    {
        return;
    }
    void* raw { nullptr };
    CHECK_CUDA(cudaMalloc(&raw, bytes));
    ptr_.reset(raw);
    bytes_ = bytes;
}

// Returns the workspace pointer passed into the cuDNN convolution.
auto CudaWorkspace::get() const -> void*
{
    return ptr_.get();
}

// Returns the size of the current allocation. cuDNN receives it together with the workspace pointer.
auto CudaWorkspace::size() const -> size_t
{
    return bytes_;
}

} // namespace dl

// Allocates the filter, bias, and gradients and sets up the cuDNN descriptors. The
// activation is the identity so the bias is added to a linear output, and LeakyReLU stays outside this layer.
Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride_val, int padding_val, float inertia_val)
    : weights_({ out_channels, in_channels, kernel_size, kernel_size }, dl::Device::GPU)
    , biases_({ 1, out_channels, 1, 1 }, dl::Device::GPU)
    , weights_gradient_({ out_channels, in_channels, kernel_size, kernel_size }, dl::Device::GPU)
    , biases_gradient_({ 1, out_channels, 1, 1 }, dl::Device::GPU)
    , in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride_val)
    , padding_(padding_val)
    , inertia_(inertia_val)
{
    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 || stride_val <= 0 || padding_val < 0)
    {
        throw std::runtime_error("Conv2d received invalid constructor arguments");
    }

    device_ = dl::Device::GPU;

    const float fan_in = static_cast<float>(in_channels_ * kernel_size_ * kernel_size_);
    const float bound = std::sqrt(dl::safe_inv(fan_in));
    fill_uniform(weights_, -bound, bound, 0xC0FFEEULL);
    fill_uniform(biases_, -bound, bound, 0xBADC0DEULL);
    fill_zero(weights_gradient_);
    fill_zero(biases_gradient_);

    const cudnnDataType_t data_type = cudnn_data_type(weights_.get_dtype());
    filter_desc_.set_nchw(out_channels_, in_channels_, kernel_size_, kernel_size_, data_type);
    conv_desc_.set_2d(padding_, stride_, CUDNN_DATA_FLOAT);
#if defined(CUDNN_TF32_TENSOR_OP_MATH)
    // TF32 when this cuDNN header defines it. An older header keeps tensor-core conversion.
    const cudnnMathType_t fp32_math = CUDNN_TF32_TENSOR_OP_MATH;
#else
    const cudnnMathType_t fp32_math = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
#endif
    // Set before any algorithm search. cuDNN picks the implementation for this math type.
    conv_desc_.set_math_type(fp32_math);
    bias_desc_.set_nchw(1, out_channels_, 1, 1, data_type);
    // Identity: convolution-with-bias stays linear. FusedCBR2d attaches LeakyReLU later.
    activation_desc_.set(CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0);
}

// Sets the input and output descriptors when the batch shape changes. The previous
// algorithm choice is dropped because the workspace and the algorithm depend on the dimensions.
auto Conv2d::configure_io_descriptors(int batch, int height, int width) -> void
{
    const std::vector<int> input_shape { batch, in_channels_, height, width };
    if (algorithms_selected_ && input_shape == input_shape_cache_)
    {
        return;
    }

    input_desc_.set_nchw(batch, in_channels_, height, width, cudnn_data_type(weights_.get_dtype()));

    int n_out { 0 };
    int c_out { 0 };
    int h_out { 0 };
    int w_out { 0 };
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc_.get(), input_desc_.get(), filter_desc_.get(), &n_out,
        &c_out, &h_out, &w_out));
    output_desc_.set_nchw(n_out, c_out, h_out, w_out, cudnn_data_type(weights_.get_dtype()));

    input_shape_cache_ = input_shape;
    output_shape_cache_ = { n_out, c_out, h_out, w_out };
    // Drop the previous algorithm choice. The workspace and the algorithm depend on these dimensions.
    algorithms_selected_ = false;
}

// Selects forward, backward-data, and backward-filter algorithms that fit the budget.
// The workspace is the maximum of the three, because one buffer serves both directions.
auto Conv2d::select_algorithms() -> void
{
    const auto handle = dl::get_cudnn_handle();
    const size_t budget = workspace_budget();
    dl::log_debug_message(std::string("Conv2d selecting cuDNN algos in=")
        + std::to_string(input_shape_cache_.empty() ? 0 : input_shape_cache_[0]) + "x"
        + std::to_string(input_shape_cache_.size() > 1 ? input_shape_cache_[1] : 0) + "x"
        + std::to_string(input_shape_cache_.size() > 2 ? input_shape_cache_[2] : 0) + "x"
        + std::to_string(input_shape_cache_.size() > 3 ? input_shape_cache_[3] : 0) + " out="
        + dl::format_shape(output_shape_cache_) + " workspace_budget="
        + std::to_string(budget / (1024U * 1024U)) + " MiB");

    cudnnConvolutionFwdAlgoPerf_t fwd_perfs[kMaxAlgoResults] {};
    int fwd_count { 0 };
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc_.get(), filter_desc_.get(), conv_desc_.get(),
        output_desc_.get(), kMaxAlgoResults, &fwd_count, fwd_perfs));
    const auto* fwd_perf = pick_perf(fwd_perfs, fwd_count, budget);
    if (fwd_perf == nullptr)
    {
        throw std::runtime_error("Conv2d: no valid cuDNN forward algorithm");
    }
    fwd_algo_ = fwd_perf->algo;

    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perfs[kMaxAlgoResults] {};
    int bwd_data_count { 0 };
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filter_desc_.get(), output_desc_.get(),
        conv_desc_.get(), input_desc_.get(), kMaxAlgoResults,
        &bwd_data_count, bwd_data_perfs));
    const auto* bwd_data_perf = pick_perf(bwd_data_perfs, bwd_data_count, budget);
    if (bwd_data_perf == nullptr)
    {
        throw std::runtime_error("Conv2d: no valid cuDNN backward-data algorithm");
    }
    bwd_data_algo_ = bwd_data_perf->algo;

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perfs[kMaxAlgoResults] {};
    int bwd_filter_count { 0 };
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, input_desc_.get(), output_desc_.get(),
        conv_desc_.get(), filter_desc_.get(), kMaxAlgoResults,
        &bwd_filter_count, bwd_filter_perfs));
    const auto* bwd_filter_perf = pick_perf(bwd_filter_perfs, bwd_filter_count, budget);
    if (bwd_filter_perf == nullptr)
    {
        throw std::runtime_error("Conv2d: no valid cuDNN backward-filter algorithm");
    }
    bwd_filter_algo_ = bwd_filter_perf->algo;

    size_t fwd_bytes { 0 };
    size_t bwd_data_bytes { 0 };
    size_t bwd_filter_bytes { 0 };
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc_.get(), filter_desc_.get(), conv_desc_.get(),
        output_desc_.get(), fwd_algo_, &fwd_bytes));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc_.get(), output_desc_.get(),
        conv_desc_.get(), input_desc_.get(), bwd_data_algo_,
        &bwd_data_bytes));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc_.get(), output_desc_.get(),
        conv_desc_.get(), filter_desc_.get(), bwd_filter_algo_,
        &bwd_filter_bytes));
    // One workspace serves forward and both backwards, so take the largest of the three.
    ensure_workspace(std::max({ fwd_bytes, bwd_data_bytes, bwd_filter_bytes }));
    algorithms_selected_ = true;
    dl::log_debug_message(std::string("Conv2d algos ready fwd=") + std::to_string(static_cast<int>(fwd_algo_))
        + " bwd_data=" + std::to_string(static_cast<int>(bwd_data_algo_)) + " bwd_filter="
        + std::to_string(static_cast<int>(bwd_filter_algo_)) + " workspace="
        + std::to_string(workspace_.size() / (1024U * 1024U)) + " MiB");
}

// Resizes the workspace to the size reported by cuDNN. The layer does not keep
// three separate allocations for the forward and the two backwards.
auto Conv2d::ensure_workspace(size_t bytes) -> void
{
    workspace_.ensure(bytes);
}

// Checks NCHW, configures the descriptors, and runs the convolution into the output
// cache. ensure does not allocate until the output shape or dtype changes.
auto Conv2d::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    require_gpu_nchw(input_tensor, "Conv2d::forward input");
    if (input_tensor.get_shape()[1] != in_channels_)
    {
        throw std::runtime_error("Conv2d::forward input channel count does not match the layer");
    }
    configure_io_descriptors(input_tensor.get_shape()[0], input_tensor.get_shape()[2], input_tensor.get_shape()[3]);
    dl::Tensor& output = dl::Tensor::ensure(output_cache_, output_shape_cache_, dl::Device::GPU, weights_.get_dtype());
    forward_into(input_tensor, output, stream);
    return output.as_view();
}

// Computes convolution-with-bias into the given buffer and caches the input for
// backward. When the algorithm does not support fusion with bias, the convolution and the bias add run separately.
auto Conv2d::forward_into(const dl::Tensor& input_tensor, dl::Tensor& output, cudaStream_t stream) -> void
{
    const dl::NvtxRange nvtx_range("Conv2d_Forward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    require_gpu_nchw(input_tensor, "Conv2d::forward input");
    if (input_tensor.get_shape()[1] != in_channels_)
    {
        throw std::runtime_error("Conv2d::forward input channel count does not match the layer");
    }

    const int batch = input_tensor.get_shape()[0];
    const int height = input_tensor.get_shape()[2];
    const int width = input_tensor.get_shape()[3];
    configure_io_descriptors(batch, height, width);
    if (!algorithms_selected_)
    {
        select_algorithms();
    }

    if (output.get_shape() != output_shape_cache_ || output.get_dtype() != weights_.get_dtype()
        || output.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error("Conv2d::forward_into output buffer does not match the convolution output");
    }

    if (input_tensor.get_dtype() != weights_.get_dtype())
    {
        input_cache_ = input_tensor.to_dtype(weights_.get_dtype(), stream);
    }
    else
    {
        input_cache_ = input_tensor.as_view();
    }
    input_cache_ready_ = true;
    const dl::Tensor& input = *input_cache_;

    const float alpha { 1.0F };
    const float alpha2_zero { 0.0F };
    const float beta_zero { 0.0F };
    const float beta_one { 1.0F };
    const auto handle = dl::get_cudnn_handle();

    const cudnnStatus_t fused_status = cudnnConvolutionBiasActivationForward(handle, &alpha, input_desc_.get(), input.data(),
        filter_desc_.get(), weights_.data(), conv_desc_.get(), fwd_algo_, workspace_.get(), workspace_.size(),
        &alpha2_zero, output_desc_.get(), output.data(), bias_desc_.get(), biases_.data(), activation_desc_.get(),
        output_desc_.get(), output.data());
    // Not every algorithm can fuse the bias. Fall back to convolution, then add the bias.
    if (fused_status != CUDNN_STATUS_SUCCESS)
    {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, input_desc_.get(), input.data(), filter_desc_.get(),
            weights_.data(), conv_desc_.get(), fwd_algo_, workspace_.get(), workspace_.size(), &beta_zero,
            output_desc_.get(), output.data()));
        CHECK_CUDNN(cudnnAddTensor(handle, &alpha, bias_desc_.get(), biases_.data(), &beta_one, output_desc_.get(),
            output.data()));
    }
}

// Computes dX and accumulates dW and db. inertia_ is the beta of the weight-gradient
// accumulation, and dX is overwritten because the input-gradient cache is not summed across calls.
auto Conv2d::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Conv2d_Backward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    if (!input_cache_ready_ || !input_cache_.has_value())
    {
        throw std::runtime_error("Conv2d::backward requires a preceding forward pass");
    }
    require_gpu_nchw(output_error_derivative, "Conv2d::backward grad_output");
    if (output_error_derivative.get_shape() != output_shape_cache_)
    {
        throw std::runtime_error("Conv2d::backward grad_output shape does not match the cached forward output");
    }
    if (!algorithms_selected_)
    {
        select_algorithms();
    }

    const dl::Tensor* grad_output = &output_error_derivative;
    dl::Tensor converted_grad;
    if (output_error_derivative.get_dtype() != weights_.get_dtype())
    {
        converted_grad = output_error_derivative.to_dtype(weights_.get_dtype(), stream);
        grad_output = &converted_grad;
    }

    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, input_cache_->get_shape(), dl::Device::GPU,
        input_cache_->get_dtype());
    const float alpha { 1.0F };
    const float beta_zero { 0.0F };
    // inertia_ accumulates dW and db. dX uses beta_zero and is overwritten.
    const float beta_momentum { inertia_ };
    const auto handle = dl::get_cudnn_handle();

    CHECK_CUDNN(cudnnConvolutionBackwardData(handle, &alpha, filter_desc_.get(), weights_.data(), output_desc_.get(),
        grad_output->data(), conv_desc_.get(), bwd_data_algo_,
        workspace_.get(), workspace_.size(), &beta_zero, input_desc_.get(),
        grad_input.data()));
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(
        handle, &alpha, input_desc_.get(), input_cache_->data(), output_desc_.get(), grad_output->data(),
        conv_desc_.get(), bwd_filter_algo_, workspace_.get(), workspace_.size(), &beta_momentum, filter_desc_.get(),
        weights_gradient_.data()));
    CHECK_CUDNN(cudnnConvolutionBackwardBias(handle, &alpha, output_desc_.get(), grad_output->data(),
        &beta_momentum, bias_desc_.get(), biases_gradient_.data()));

    input_cache_ready_ = false;
    return grad_input.as_view();
}

// Updates the weights and bias with SGD, and when momentum > 0 keeps velocity in an
// optional buffer. A frozen layer returns immediately and does not touch the filter.
void Conv2d::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("Conv2d_Step");
    const dl::StreamGuard stream_guard(stream);
    if (frozen())
    {
        return;
    }
    const float clip = parameter_clip_bound();
    const float lr = step_learning_rate();
    if (momentum > 0.0F)
    {
        dl::Tensor& weight_velocity = ensure_zero_like(weights_velocity_, weights_);
        dl::Tensor& bias_velocity = ensure_zero_like(biases_velocity_, biases_);
        weights_.sgd_momentum_update_(weights_gradient_, weight_velocity, lr, momentum, weight_decay, clip);
        biases_.sgd_momentum_update_(biases_gradient_, bias_velocity, lr, momentum, weight_decay, clip);
        return;
    }
    weights_.sgd_update_(weights_gradient_, lr, weight_decay, clip);
    biases_.sgd_update_(biases_gradient_, lr, weight_decay, clip);
}

// Clips the filter and bias gradients to a symmetric range. A bound <= 0 turns clipping off.
void Conv2d::clip_gradients(float abs_bound, cudaStream_t stream)
{
    const dl::StreamGuard stream_guard(stream);
    if (abs_bound <= 0.0F)
    {
        return;
    }
    weights_gradient_.clamp_(-abs_bound, abs_bound);
    biases_gradient_.clamp_(-abs_bound, abs_bound);
}

// Returns views of the filter and bias for saving the network. Gradients and
// momentum velocity are not written to the file.
auto Conv2d::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights", weights_.view(weights_.get_shape()));
    params.emplace("bias", biases_.view(biases_.get_shape()));
    return params;
}

// Loads the filter and bias with a device-to-device copy. The buffer shapes stay as they were at construction.
void Conv2d::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    copy_same_size(weights_, params.at("weights"), "Conv2d::set_parameters weights");
    copy_same_size(biases_, params.at("bias"), "Conv2d::set_parameters bias");
}

// Leaves the parameters on the GPU. The cuDNN convolution has no host path.
auto Conv2d::to(dl::Device device) -> void
{
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("Conv2d parameters must remain on the GPU");
    }
    device_ = device;
}
