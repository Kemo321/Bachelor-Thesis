# Conv2d.cpp

2D cuDNN convolution with bias, algorithm selection, and an SGD step.

## fill_uniform

Fills the tensor on the host with `paired_uniform`, then copies once to the device. A device evaluation of the same formula is contracted by nvcc and is not bit-identical to the host values copied into LibTorch. An empty tensor is skipped.

## fill_zero

Zeros the buffer on the current stream. Gradients start at zero and do not need a separate fill kernel.

## ensure_zero_like

Creates or resizes the velocity buffer and zeros it when the shape changes. SGD momentum needs a velocity with the weight's shape; a fresh allocation left unzeroed would be garbage.

## require_gpu_nchw

Rejects a tensor that is off the GPU or not rank 4. The cuDNN convolution reads NCHW layout from the descriptor.

## copy_same_size

Copies weights device-to-device, converting on the current stream when the dtype differs. The store must not change the filter shape.

## workspace_budget

Returns the cuDNN workspace budget. A reserve is left so the chosen algorithm does not take all free device memory and fail the next allocation.

## supports_bias_activation

True for the forward algorithms that can add bias inside `cudnnConvolutionBiasActivationForward`: implicit GEMM, implicit precomputed GEMM, and Winograd.

## pick_fastest

Returns the successful result with the lowest measured time that fits the workspace budget. Both `cudnnFind` and the v7 heuristic fill `time`.

## pick_forward

Starts from `pick_fastest`. When that algorithm cannot fuse the bias add, and a bias-capable algorithm is within `1.25 * fastest + 0.02` ms, that one is used instead.

## CudnnContext::CudnnContext()

Creates the process cuDNN handle. One context serves every layer, so cudnnCreate is not called on every convolution.

## CudnnContext::~CudnnContext

Destroys the cuDNN handle if construction succeeded. The destructor must not throw, so a cuDNN error here is deliberately ignored.

## CudnnContext::handle

Returns the handle from a function-local static instance. The first call creates the context; later calls only return the same handle.

## get_cudnn_handle

Hands the shared cuDNN handle to the layers. A layer does not keep its own copy of the context.

## CudnnTensorDescriptor::CudnnTensorDescriptor()

Creates a cuDNN tensor descriptor. The wrapper lives with the layer, so forward only replaces the dimensions, not the handle itself.

## CudnnTensorDescriptor::~CudnnTensorDescriptor

Destroys the tensor descriptor if construction succeeded. The destructor must not throw, so a cuDNN error here is deliberately ignored.

## CudnnTensorDescriptor::CudnnTensorDescriptor(CudnnTensorDescriptor&&)

Takes ownership of the tensor descriptor and nulls the source. Otherwise the source destructor would free the same handle a second time.

## CudnnTensorDescriptor::operator=

Releases this tensor descriptor and takes the other one. The identity check guards against a double destroy on self-assignment.

## CudnnTensorDescriptor::get

Returns the raw tensor-descriptor handle for cuDNN calls.

## CudnnTensorDescriptor::set_nchw

Writes NCHW layout and the data type into the descriptor. Convolution reads the dimensions from here, not from the tensor metadata.

## CudnnFilterDescriptor::CudnnFilterDescriptor()

Creates a cuDNN filter descriptor. The kernel shape is fixed, so the handle is created once.

## CudnnFilterDescriptor::~CudnnFilterDescriptor

Destroys the filter descriptor if construction succeeded. The destructor must not throw, so a cuDNN error here is deliberately ignored.

## CudnnFilterDescriptor::CudnnFilterDescriptor(CudnnFilterDescriptor&&)

Takes ownership of the filter descriptor and nulls the source. Otherwise the source destructor would free the same handle a second time.

## CudnnFilterDescriptor::operator=

Releases this filter descriptor and takes the other one. The identity check guards against a double destroy on self-assignment.

## CudnnFilterDescriptor::get

Returns the raw filter-descriptor handle for cuDNN calls.

## CudnnFilterDescriptor::set_nchw

Writes the NCHW kernel shape and the data type. Output channels are the K dimension of the cuDNN filter.

## CudnnConvolutionDescriptor::CudnnConvolutionDescriptor()

Creates a cuDNN convolution descriptor. Padding and stride are fixed for the layer, so the handle is not created on every forward.

## CudnnConvolutionDescriptor::~CudnnConvolutionDescriptor

Destroys the convolution descriptor if construction succeeded. The destructor must not throw, so a cuDNN error here is deliberately ignored.

## CudnnConvolutionDescriptor::CudnnConvolutionDescriptor(CudnnConvolutionDescriptor&&)

Takes ownership of the convolution descriptor and nulls the source. Otherwise the source destructor would free the same handle a second time.

## CudnnConvolutionDescriptor::operator=

Releases this convolution descriptor and takes the other one. The identity check guards against a double destroy on self-assignment.

## CudnnConvolutionDescriptor::get

Returns the raw convolution-descriptor handle for cuDNN calls.

## CudnnConvolutionDescriptor::set_2d

Sets padding, stride, cross-correlation, and the accumulation type. Dilation stays 1 because the layer does not expose a kernel-dilation parameter.

## CudnnConvolutionDescriptor::set_math_type

Selects the convolution math mode, for example TF32. cuDNN picks an implementation for that mode, so it must be set before the algorithm search.

## CudnnActivationDescriptor::CudnnActivationDescriptor()

Creates a cuDNN activation descriptor. Convolution-with-bias needs one even when the activation is the identity.

## CudnnActivationDescriptor::~CudnnActivationDescriptor

Destroys the activation descriptor if construction succeeded. The destructor must not throw, so a cuDNN error here is deliberately ignored.

## CudnnActivationDescriptor::CudnnActivationDescriptor(CudnnActivationDescriptor&&)

Takes ownership of the activation descriptor and nulls the source. Otherwise the source destructor would free the same handle a second time.

## CudnnActivationDescriptor::operator=

Releases this activation descriptor and takes the other one. The identity check guards against a double destroy on self-assignment.

## CudnnActivationDescriptor::get

Returns the raw activation-descriptor handle for cuDNN calls.

## CudnnActivationDescriptor::set

Writes the activation mode, the NaN policy, and the coefficient. Conv2d sets the identity so convolution-with-bias stays linear.

## CudaWorkspace::Deleter::operator()

Frees the cuDNN workspace buffer. The deleter sits in a unique_ptr so cudaFree runs when the workspace dies, rather than by hand on every ensure.

## CudaWorkspace::ensure

Grows the workspace when cuDNN reports a larger size. A smaller request keeps the old allocation so memory is not freed on every forward.

## CudaWorkspace::get

Returns the workspace pointer passed into the cuDNN convolution.

## CudaWorkspace::size

Returns the size of the current allocation. cuDNN receives it together with the workspace pointer.

## Conv2d::Conv2d

Allocates the filter, bias, and gradients and sets up the cuDNN descriptors. The activation is the identity so the bias is added to a linear output, and LeakyReLU stays outside this layer.

## Conv2d::configure_io_descriptors

Sets the input and output descriptors when the batch shape changes. The previous algorithm choice is dropped because the workspace and the algorithm depend on the dimensions.

## Conv2d::select_algorithms

Times up to ten algorithms with `cudnnFind` for forward, backward-data, and backward-filter. A failed find, or an empty list, falls back to the v7 heuristic. Forward uses `pick_forward`; if that still returns nothing, v7 is queried once more. The other two directions use `pick_fastest`. One workspace is sized to the maximum of the three reported sizes.

## Conv2d::ensure_workspace

Resizes the workspace to the size reported by cuDNN. The layer does not keep three separate allocations for the forward and the two backwards.

## Conv2d::forward

Checks NCHW, configures the descriptors, and runs the convolution into the output cache. ensure does not allocate until the output shape or dtype changes.

## Conv2d::forward_into

Computes convolution-with-bias into the given buffer and caches the input for backward. When the algorithm does not support fusion with bias, the convolution and the bias add run separately.

## Conv2d::backward

Computes dX and accumulates dW and db. inertia_ is the beta of the weight-gradient accumulation, and dX is overwritten because the input-gradient cache is not summed across calls.

## Conv2d::step

Updates the weights and bias with SGD, and when momentum > 0 keeps velocity in an optional buffer. A frozen layer returns immediately and does not touch the filter.

## Conv2d::clip_gradients

Clips the filter and bias gradients to a symmetric range. A bound <= 0 turns clipping off.

## Conv2d::get_parameters

Returns views of the filter and bias for saving the network. Gradients and momentum velocity are not written to the file.

## Conv2d::set_parameters

Loads the filter and bias with a device-to-device copy. The buffer shapes stay as they were at construction.

## Conv2d::to

Leaves the parameters on the GPU. The cuDNN convolution has no host path.
