# MaxPool2d.cpp

2D max-pooling through cuDNN, with the input and output cached for backward.

## CudnnPoolingDescriptor::CudnnPoolingDescriptor()

Creates a cuDNN pooling descriptor. The layer keeps it for its whole lifetime, so cudnnCreate is not called on every forward.

## CudnnPoolingDescriptor::~CudnnPoolingDescriptor

Destroys the descriptor if construction succeeded. The destructor must not throw, so a cuDNN error here is deliberately ignored.

## CudnnPoolingDescriptor::CudnnPoolingDescriptor(CudnnPoolingDescriptor&&)

Takes ownership of the descriptor and nulls the source. Otherwise the source destructor would free the same handle a second time.

## CudnnPoolingDescriptor::operator=

Releases this descriptor and takes the other one. The identity check guards against a double destroy on self-assignment.

## CudnnPoolingDescriptor::get

Returns the raw handle for cuDNN calls. The layer does not create a second descriptor for the duration of one call.

## CudnnPoolingDescriptor::set_max_2d

Sets a square max-pool window, stride, and padding. The mode is CUDNN_POOLING_MAX because the layer does not compute average pooling.

## require_gpu_nchw

Rejects a tensor that is off the GPU or not rank 4. cuDNN 2D pooling reads NCHW layout from the descriptor, not from the tensor metadata.

## MaxPool2d::MaxPool2d

Checks that the kernel and stride are positive, then writes them into the pooling descriptor. The window geometry is fixed; only the input shape changes on forward.

## MaxPool2d::configure_descriptors

Builds the input and output descriptors and asks cuDNN for the pooled size. The output shape is not computed by hand, so padding and stride stay consistent with forward.

## MaxPool2d::forward

Runs cudnnPoolingForward and caches the input and the output. Max-pool backward needs both to recover which cell in the window won.

## MaxPool2d::backward

Routes the gradient only to the winning cells of each window through cudnnPoolingBackward. The cache is then cleared, because the argmax belongs to that forward.
