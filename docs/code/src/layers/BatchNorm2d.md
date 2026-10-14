# BatchNorm2d.cpp

Spatial cuDNN batch-norm: batch statistics in train, running mean and var in eval.

## fill_constant_kernel

Writes a constant into a GPU buffer. A separate kernel, because cudaMemset can only write zero.

## fill_constant

Zeros the buffer with cudaMemsetAsync, and writes any other constant with the kernel. Gamma and the running variance start at 1, so memset alone is not enough.

## require_gpu_nchw

Rejects a tensor that is off the GPU or not rank 4. Spatial cuDNN batch-norm reads NCHW layout from the descriptor.

## copy_same_size

Copies a parameter device-to-device when the sizes match. Loading weights must not change the shape of the layer buffers.

## batchnorm_channel_shape

Returns the shape [1, C, 1, 1] and checks the channel count and eps. That layout matches the cuDNN spatial batch-norm descriptor.

## BatchNorm2d::BatchNorm2d

Allocates gamma, beta, the running statistics, and the save buffers on the GPU. Gamma starts at 1 and the running variance starts at 1, so the first inference does not divide by zero.

## BatchNorm2d::configure_descriptors

Sets the NCHW descriptor and derives the spatial BN descriptor from it. Derive, rather than a hand-written shape, keeps the gamma scale consistent with CUDNN_BATCHNORM_SPATIAL.

## BatchNorm2d::forward

In train, computes batch statistics and updates running mean/var; in eval, normalises with the saved statistics. The input is cached only in train, because backward reads x and save_*.

## BatchNorm2d::backward

Computes dx, dgamma, and dbeta from the statistics saved by the training forward. In eval, backward has neither the input nor save_mean / save_inv_var.

## BatchNorm2d::step

Takes an SGD step on gamma and beta. Running mean and variance are not touched here: cuDNN already writes them in the training forward.

## BatchNorm2d::clip_gradients

Clips the gamma and beta gradients to a symmetric range. A bound <= 0 turns clipping off, as parameter_clip_bound does.

## BatchNorm2d::get_parameters

Returns views of gamma, beta, and the running statistics for saving the network. Gradients and the save buffers are not written to the weight file.

## BatchNorm2d::set_parameters

Loads the saved tensors into the layer buffers with a device-to-device copy. The shape stays as it was at construction; only the contents are copied.

## BatchNorm2d::to

Leaves the parameters on the GPU. The layer has no host path for the cuDNN calls.
