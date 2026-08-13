#include "DeepLearnLib/Conv2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace
{

struct UniformFill
{
    float low;
    float high;
    unsigned long long seed;

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

struct ScaledAdd
{
    float scale;

    __host__ __device__ auto operator()(float lhs, float rhs) const -> float
    {
        return lhs + (scale * rhs);
    }
};

auto fill_uniform(dl::Tensor& tensor, float low, float high, unsigned long long seed) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    auto out = thrust::device_pointer_cast(tensor.data());
    thrust::transform(thrust::device, thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(static_cast<int>(tensor.get_size())), out,
                      UniformFill{ low, high, seed });
    CHECK_CUDA(cudaGetLastError());
}

auto fill_zero(dl::Tensor& tensor) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    auto out = thrust::device_pointer_cast(tensor.data());
    thrust::fill(thrust::device, out, out + static_cast<std::ptrdiff_t>(tensor.get_size()), 0.0F);
    CHECK_CUDA(cudaGetLastError());
}

auto add_scaled(dl::Tensor& lhs, const dl::Tensor& rhs, float scale) -> void
{
    if (lhs.get_size() == 0)
    {
        return;
    }
    auto dest = thrust::device_pointer_cast(lhs.data());
    auto src = thrust::device_pointer_cast(rhs.data());
    thrust::transform(thrust::device, dest, dest + static_cast<std::ptrdiff_t>(lhs.get_size()), src, dest,
                      ScaledAdd{ scale });
    CHECK_CUDA(cudaGetLastError());
}

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
    CHECK_CUDA(cudaMemcpy(dst.data(), src.data(), src.get_size() * sizeof(float), cudaMemcpyDeviceToDevice));
}

constexpr float kWeightDecay = 0.0005F;
constexpr int kMaxAlgoResults = 10;

auto workspace_budget() -> size_t
{
    size_t free_bytes{ 0 };
    size_t total_bytes{ 0 };
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    constexpr size_t kReserve = 64ULL * 1024ULL * 1024ULL;
    if (free_bytes > kReserve)
    {
        return free_bytes - kReserve;
    }
    return free_bytes / 2U;
}

template <typename PerfT>
auto pick_perf(const PerfT* perfs, int count, size_t budget) -> const PerfT*
{
    for (int idx = 0; idx < count; ++idx)
    {
        if (perfs[idx].status == CUDNN_STATUS_SUCCESS && perfs[idx].memory <= budget)
        {
            return &perfs[idx];
        }
    }
    return nullptr;
}

} // namespace

namespace dl
{

CudnnContext::CudnnContext()
{
    CHECK_CUDNN(cudnnCreate(&handle_));
}

CudnnContext::~CudnnContext()
{
    if (handle_ != nullptr)
    {
        static_cast<void>(cudnnDestroy(handle_));
        handle_ = nullptr;
    }
}

auto CudnnContext::handle() -> cudnnHandle_t
{
    static CudnnContext context;
    return context.handle_;
}

auto get_cudnn_handle() -> cudnnHandle_t
{
    return CudnnContext::handle();
}

CudnnTensorDescriptor::CudnnTensorDescriptor()
{
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc_));
}

CudnnTensorDescriptor::~CudnnTensorDescriptor()
{
    if (desc_ != nullptr)
    {
        static_cast<void>(cudnnDestroyTensorDescriptor(desc_));
    }
}

CudnnTensorDescriptor::CudnnTensorDescriptor(CudnnTensorDescriptor&& other) noexcept
    : desc_(other.desc_)
{
    other.desc_ = nullptr;
}

auto CudnnTensorDescriptor::operator=(CudnnTensorDescriptor&& other) noexcept -> CudnnTensorDescriptor&
{
    if (this != &other)
    {
        if (desc_ != nullptr)
        {
            static_cast<void>(cudnnDestroyTensorDescriptor(desc_));
        }
        desc_ = other.desc_;
        other.desc_ = nullptr;
    }
    return *this;
}

auto CudnnTensorDescriptor::get() const -> cudnnTensorDescriptor_t
{
    return desc_;
}

auto CudnnTensorDescriptor::set_nchw(int n, int c, int h, int w) -> void
{
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
}

CudnnFilterDescriptor::CudnnFilterDescriptor()
{
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&desc_));
}

CudnnFilterDescriptor::~CudnnFilterDescriptor()
{
    if (desc_ != nullptr)
    {
        static_cast<void>(cudnnDestroyFilterDescriptor(desc_));
    }
}

CudnnFilterDescriptor::CudnnFilterDescriptor(CudnnFilterDescriptor&& other) noexcept
    : desc_(other.desc_)
{
    other.desc_ = nullptr;
}

auto CudnnFilterDescriptor::operator=(CudnnFilterDescriptor&& other) noexcept -> CudnnFilterDescriptor&
{
    if (this != &other)
    {
        if (desc_ != nullptr)
        {
            static_cast<void>(cudnnDestroyFilterDescriptor(desc_));
        }
        desc_ = other.desc_;
        other.desc_ = nullptr;
    }
    return *this;
}

auto CudnnFilterDescriptor::get() const -> cudnnFilterDescriptor_t
{
    return desc_;
}

auto CudnnFilterDescriptor::set_nchw(int out_channels, int in_channels, int kernel_h, int kernel_w) -> void
{
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels,
                                           kernel_h, kernel_w));
}

CudnnConvolutionDescriptor::CudnnConvolutionDescriptor()
{
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&desc_));
}

CudnnConvolutionDescriptor::~CudnnConvolutionDescriptor()
{
    if (desc_ != nullptr)
    {
        static_cast<void>(cudnnDestroyConvolutionDescriptor(desc_));
    }
}

CudnnConvolutionDescriptor::CudnnConvolutionDescriptor(CudnnConvolutionDescriptor&& other) noexcept
    : desc_(other.desc_)
{
    other.desc_ = nullptr;
}

auto CudnnConvolutionDescriptor::operator=(CudnnConvolutionDescriptor&& other) noexcept -> CudnnConvolutionDescriptor&
{
    if (this != &other)
    {
        if (desc_ != nullptr)
        {
            static_cast<void>(cudnnDestroyConvolutionDescriptor(desc_));
        }
        desc_ = other.desc_;
        other.desc_ = nullptr;
    }
    return *this;
}

auto CudnnConvolutionDescriptor::get() const -> cudnnConvolutionDescriptor_t
{
    return desc_;
}

auto CudnnConvolutionDescriptor::set_2d(int padding, int stride) -> void
{
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(desc_, padding, padding, stride, stride, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
}

void CudaWorkspace::Deleter::operator()(void* pointer) const
{
    if (pointer != nullptr)
    {
        static_cast<void>(cudaFree(pointer));
    }
}

auto CudaWorkspace::ensure(size_t bytes) -> void
{
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
    void* raw{ nullptr };
    CHECK_CUDA(cudaMalloc(&raw, bytes));
    ptr_.reset(raw);
    bytes_ = bytes;
}

auto CudaWorkspace::get() const -> void*
{
    return ptr_.get();
}

auto CudaWorkspace::size() const -> size_t
{
    return bytes_;
}

} // namespace dl

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
    filter_desc_.set_nchw(out_channels_, in_channels_, kernel_size_, kernel_size_);
    conv_desc_.set_2d(padding_, stride_);
    bias_desc_.set_nchw(1, out_channels_, 1, 1);

    const float fan_in = static_cast<float>(in_channels_ * kernel_size_ * kernel_size_);
    const float bound = std::sqrt(1.0F / fan_in);
    fill_uniform(weights_, -bound, bound, 0xC0FFEEULL);
    fill_uniform(biases_, -bound, bound, 0xBADC0DEULL);
    fill_zero(weights_gradient_);
    fill_zero(biases_gradient_);
}

auto Conv2d::configure_io_descriptors(int batch, int height, int width) -> void
{
    const std::vector<int> input_shape{ batch, in_channels_, height, width };
    if (algorithms_selected_ && input_shape == input_shape_cache_)
    {
        return;
    }

    input_desc_.set_nchw(batch, in_channels_, height, width);

    int n_out{ 0 };
    int c_out{ 0 };
    int h_out{ 0 };
    int w_out{ 0 };
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc_.get(), input_desc_.get(), filter_desc_.get(), &n_out,
                                                      &c_out, &h_out, &w_out));
    output_desc_.set_nchw(n_out, c_out, h_out, w_out);

    input_shape_cache_ = input_shape;
    output_shape_cache_ = { n_out, c_out, h_out, w_out };
    algorithms_selected_ = false;
}

auto Conv2d::select_algorithms() -> void
{
    const auto handle = dl::get_cudnn_handle();
    const size_t budget = workspace_budget();

    cudnnConvolutionFwdAlgoPerf_t fwd_perfs[kMaxAlgoResults]{};
    int fwd_count{ 0 };
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc_.get(), filter_desc_.get(), conv_desc_.get(),
                                                       output_desc_.get(), kMaxAlgoResults, &fwd_count, fwd_perfs));
    const auto* fwd_perf = pick_perf(fwd_perfs, fwd_count, budget);
    if (fwd_perf == nullptr)
    {
        throw std::runtime_error("Conv2d: no valid cuDNN forward algorithm");
    }
    fwd_algo_ = fwd_perf->algo;

    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perfs[kMaxAlgoResults]{};
    int bwd_data_count{ 0 };
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filter_desc_.get(), output_desc_.get(),
                                                            conv_desc_.get(), input_desc_.get(), kMaxAlgoResults,
                                                            &bwd_data_count, bwd_data_perfs));
    const auto* bwd_data_perf = pick_perf(bwd_data_perfs, bwd_data_count, budget);
    if (bwd_data_perf == nullptr)
    {
        throw std::runtime_error("Conv2d: no valid cuDNN backward-data algorithm");
    }
    bwd_data_algo_ = bwd_data_perf->algo;

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perfs[kMaxAlgoResults]{};
    int bwd_filter_count{ 0 };
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, input_desc_.get(), output_desc_.get(),
                                                              conv_desc_.get(), filter_desc_.get(), kMaxAlgoResults,
                                                              &bwd_filter_count, bwd_filter_perfs));
    const auto* bwd_filter_perf = pick_perf(bwd_filter_perfs, bwd_filter_count, budget);
    if (bwd_filter_perf == nullptr)
    {
        throw std::runtime_error("Conv2d: no valid cuDNN backward-filter algorithm");
    }
    bwd_filter_algo_ = bwd_filter_perf->algo;

    size_t fwd_bytes{ 0 };
    size_t bwd_data_bytes{ 0 };
    size_t bwd_filter_bytes{ 0 };
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc_.get(), filter_desc_.get(), conv_desc_.get(),
                                                        output_desc_.get(), fwd_algo_, &fwd_bytes));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc_.get(), output_desc_.get(),
                                                             conv_desc_.get(), input_desc_.get(), bwd_data_algo_,
                                                             &bwd_data_bytes));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc_.get(), output_desc_.get(),
                                                               conv_desc_.get(), filter_desc_.get(), bwd_filter_algo_,
                                                               &bwd_filter_bytes));
    ensure_workspace(std::max({ fwd_bytes, bwd_data_bytes, bwd_filter_bytes }));
    algorithms_selected_ = true;
}

auto Conv2d::ensure_workspace(size_t bytes) -> void
{
    workspace_.ensure(bytes);
}

auto Conv2d::forward(const dl::Tensor& input_tensor) -> dl::Tensor
{
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

    input_cache_ = dl::Tensor(input_tensor.get_shape(), dl::Device::GPU);
    copy_same_size(*input_cache_, input_tensor, "Conv2d::forward input cache");

    dl::Tensor output(output_shape_cache_, dl::Device::GPU);
    const float alpha{ 1.0F };
    const float beta_zero{ 0.0F };
    const float beta_one{ 1.0F };
    const auto handle = dl::get_cudnn_handle();

    CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, input_desc_.get(), input_tensor.data(), filter_desc_.get(),
                                        weights_.data(), conv_desc_.get(), fwd_algo_, workspace_.get(),
                                        workspace_.size(), &beta_zero, output_desc_.get(), output.data()));
    CHECK_CUDNN(cudnnAddTensor(handle, &alpha, bias_desc_.get(), biases_.data(), &beta_one, output_desc_.get(),
                               output.data()));
    return output;
}

auto Conv2d::backward(const dl::Tensor& output_error_derivative) -> dl::Tensor
{
    if (!input_cache_.has_value())
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

    dl::Tensor grad_input(input_cache_->get_shape(), dl::Device::GPU);
    const float alpha{ 1.0F };
    const float beta_zero{ 0.0F };
    const float beta_momentum{ inertia_ };
    const auto handle = dl::get_cudnn_handle();

    CHECK_CUDNN(cudnnConvolutionBackwardData(handle, &alpha, filter_desc_.get(), weights_.data(), output_desc_.get(),
                                             output_error_derivative.data(), conv_desc_.get(), bwd_data_algo_,
                                             workspace_.get(), workspace_.size(), &beta_zero, input_desc_.get(),
                                             grad_input.data()));
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(
        handle, &alpha, input_desc_.get(), input_cache_->data(), output_desc_.get(), output_error_derivative.data(),
        conv_desc_.get(), bwd_filter_algo_, workspace_.get(), workspace_.size(), &beta_momentum, filter_desc_.get(),
        weights_gradient_.data()));
    CHECK_CUDNN(cudnnConvolutionBackwardBias(handle, &alpha, output_desc_.get(), output_error_derivative.data(),
                                             &beta_momentum, bias_desc_.get(), biases_gradient_.data()));

    add_scaled(weights_gradient_, weights_, kWeightDecay);
    add_scaled(biases_gradient_, biases_, kWeightDecay);

    input_cache_.reset();
    return grad_input;
}

void Conv2d::step()
{
    weights_ = weights_ - (weights_gradient_ * learning_rate);
    biases_ = biases_ - (biases_gradient_ * learning_rate);
}

auto Conv2d::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights", weights_.view(weights_.get_shape()));
    params.emplace("bias", biases_.view(biases_.get_shape()));
    return params;
}

void Conv2d::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    copy_same_size(weights_, params.at("weights"), "Conv2d::set_parameters weights");
    copy_same_size(biases_, params.at("bias"), "Conv2d::set_parameters bias");
}

auto Conv2d::to(dl::Device device) -> void
{
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("Conv2d parameters must remain on the GPU");
    }
    device_ = device;
}
