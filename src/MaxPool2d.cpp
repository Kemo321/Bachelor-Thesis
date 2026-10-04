#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Nvtx.hpp"

#include <stdexcept>
#include <string>

namespace dl
{

// Creates a cuDNN pooling descriptor. The layer keeps it for its whole lifetime,
// so cudnnCreate is not called on every forward.
CudnnPoolingDescriptor::CudnnPoolingDescriptor()
{
    // Created once with the layer. Forward does not allocate another pooling descriptor.
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&desc_));
}

// Destroys the descriptor if construction succeeded. The destructor must not throw,
// so a cuDNN error here is deliberately ignored.
CudnnPoolingDescriptor::~CudnnPoolingDescriptor()
{
    if (desc_ != nullptr)
    {
        // Ignored on purpose: a destructor must not throw.
        static_cast<void>(cudnnDestroyPoolingDescriptor(desc_));
    }
}

// Takes ownership of the descriptor and nulls the source. Otherwise the source
// destructor would free the same handle a second time.
CudnnPoolingDescriptor::CudnnPoolingDescriptor(CudnnPoolingDescriptor&& other) noexcept
    : desc_(other.desc_)
{
    // The source must not destroy the handle this object now owns.
    other.desc_ = nullptr;
}

// Releases this descriptor and takes the other one. The identity check guards
// against a double destroy on self-assignment.
auto CudnnPoolingDescriptor::operator=(CudnnPoolingDescriptor&& other) noexcept -> CudnnPoolingDescriptor&
{
    // Self-assignment would destroy the handle and then use it.
    if (this != &other)
    {
        if (desc_ != nullptr)
        {
            static_cast<void>(cudnnDestroyPoolingDescriptor(desc_));
        }
        desc_ = other.desc_;
        // The source must not destroy the handle this object now owns.
        other.desc_ = nullptr;
    }
    return *this;
}

// Returns the raw handle for cuDNN calls. The layer does not create a second
// descriptor for the duration of one call.
auto CudnnPoolingDescriptor::get() const -> cudnnPoolingDescriptor_t
{
    return desc_;
}

// Sets a square max-pool window, stride, and padding. The mode is CUDNN_POOLING_MAX
// because the layer does not compute average pooling.
auto CudnnPoolingDescriptor::set_max_2d(int window, int stride, int padding) -> void
{
    // Max, not average. The window is square: the same size is used for height and width.
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(desc_, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, window, window, padding,
        padding, stride, stride));
}

} // namespace dl

namespace
{

// Rejects a tensor that is off the GPU or not rank 4. cuDNN 2D pooling reads NCHW
// layout from the descriptor, not from the tensor metadata.
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

} // namespace

// Checks that the kernel and stride are positive, then writes them into the pooling
// descriptor. The window geometry is fixed; only the input shape changes on forward.
MaxPool2d::MaxPool2d(int kernel_size_val, int stride_val)
    : kernel_size_(kernel_size_val)
    , stride_(stride_val)
{
    if (kernel_size_val <= 0 || stride_val <= 0)
    {
        throw std::runtime_error("MaxPool2d requires positive kernel size and stride");
    }

    device_ = dl::Device::GPU;
    pooling_desc_.set_max_2d(kernel_size_, stride_);
}

// Builds the input and output descriptors and asks cuDNN for the pooled size.
// The output shape is not computed by hand, so padding and stride stay consistent with forward.
auto MaxPool2d::configure_descriptors(int batch, int channels, int height, int width, dl::Dtype dtype) -> void
{
    const std::vector<int> input_shape { batch, channels, height, width };
    input_desc_.set_nchw(batch, channels, height, width, cudnn_data_type(dtype));

    int n_out { 0 };
    int c_out { 0 };
    int h_out { 0 };
    int w_out { 0 };
    // Output size comes from cuDNN so it matches the window, padding, and stride used in forward.
    CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(pooling_desc_.get(), input_desc_.get(), &n_out, &c_out, &h_out,
        &w_out));
    output_desc_.set_nchw(n_out, c_out, h_out, w_out, cudnn_data_type(dtype));

    input_shape_cache_ = input_shape;
    output_shape_cache_ = { n_out, c_out, h_out, w_out };
    descriptors_configured_ = true;
}

// Runs cudnnPoolingForward and caches the input and the output. Max-pool backward
// needs both to recover which cell in the window won.
auto MaxPool2d::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("MaxPool2d_Forward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    require_gpu_nchw(input_tensor, "MaxPool2d::forward input");

    const int batch = input_tensor.get_shape()[0];
    const int channels = input_tensor.get_shape()[1];
    const int height = input_tensor.get_shape()[2];
    const int width = input_tensor.get_shape()[3];
    configure_descriptors(batch, channels, height, width, input_tensor.get_dtype());

    // Both caches are required later: backward finds the winning cell from x and y.
    input_cache_ = input_tensor.as_view();

    dl::Tensor& out_cached = dl::Tensor::ensure(output_cache_, output_shape_cache_, dl::Device::GPU,
        input_tensor.get_dtype());
    const float alpha { 1.0F };
    const float beta_zero { 0.0F };
    CHECK_CUDNN(cudnnPoolingForward(dl::get_cudnn_handle(), pooling_desc_.get(), &alpha, input_desc_.get(),
        input_tensor.data(), &beta_zero, output_desc_.get(), out_cached.data()));

    caches_ready_ = true;
    return out_cached.as_view();
}

// Routes the gradient only to the winning cells of each window through
// cudnnPoolingBackward. The cache is then cleared, because the argmax belongs to that forward.
auto MaxPool2d::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("MaxPool2d_Backward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    if (!caches_ready_ || !input_cache_.has_value() || !output_cache_.has_value())
    {
        throw std::runtime_error("MaxPool2d::backward requires a preceding forward pass");
    }
    require_gpu_nchw(output_error_derivative, "MaxPool2d::backward grad_output");
    if (output_error_derivative.get_shape() != output_shape_cache_)
    {
        throw std::runtime_error("MaxPool2d::backward grad_output shape does not match the cached pooling output");
    }

    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, input_cache_->get_shape(), dl::Device::GPU,
        input_cache_->get_dtype());
    const float alpha { 1.0F };
    const float beta_zero { 0.0F };
    // Only the winning cell in each window receives gradient. The cached input and output identify that cell.
    CHECK_CUDNN(cudnnPoolingBackward(dl::get_cudnn_handle(), pooling_desc_.get(), &alpha, output_desc_.get(),
        output_cache_->data(), output_desc_.get(), output_error_derivative.data(),
        input_desc_.get(), input_cache_->data(), &beta_zero, input_desc_.get(),
        grad_input.data()));

    // The argmax belongs to this forward. A later backward must not reuse it.
    caches_ready_ = false;
    return grad_input.as_view();
}
