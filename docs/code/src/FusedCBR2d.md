# FusedCBR2d.cpp

Convolution with bias, then one pass of batch-norm and LeakyReLU.

## fill_constant_kernel

Writes a constant into a GPU buffer. cudaMemset can only write zero, and gamma and the variance start at 1.

## fill_constant

Zeros the buffer with cudaMemsetAsync, and writes any other constant with the kernel. Batch-norm parameter initialisation cannot go through memset alone.

## ensure_zero_like

Creates or resizes the velocity buffer and zeros it when the shape changes. SGD momentum on gamma and beta needs a velocity; a fresh allocation left unzeroed would be garbage.

## copy_same_size

Copies a parameter device-to-device, converting on the current stream when the dtype differs. Storing weights must not change the channel count.

## require_gpu_nchw

Rejects a tensor that is off the GPU or not rank 4. Convolution and batch-norm read NCHW layout.

## load_act

Loads an activation element as float. The moments and the BN+LeakyReLU fusion compute in float regardless of the storage type.

## store_act

Stores a float result into the activation buffer. One store serves both the fusion and the LeakyReLU derivative.

## spatial_moments_kernel

One block per channel sums the values and the squares over the batch and the pixels, then reduces in shared memory. The batch mean and variance feed normalisation without global atomics.

## finalize_bn_stats_kernel

In train, computes the inverse standard deviation and folds the batch into running mean/var; in eval, substitutes the running statistics. A separate kernel, because the fusion and the cuDNN backward both read inv_std.

## fused_bn_leaky_kernel

In one pass, applies affine batch-norm and LeakyReLU. A separate kernel for the activation alone would read the whole tensor from global memory again.

## leaky_backward_from_output_kernel

Multiplies the gradient by 1 or by the slope according to the sign of the fused output. The pre-activation is not stored separately, so the LeakyReLU branch is recovered from the saved y.

## elementwise_grid

Computes the block grid of the elementwise kernel. Rounding up by the element count and kElementwiseThreads covers the last, partial block.

## channel_shape

Returns the batch-norm parameter shape [1, C, 1, 1]. That layout matches spatial batch-norm and the gamma, beta, and per-channel statistic buffers.

## FusedCBR2d::FusedCBR2d

Builds the convolution together with the batch-norm buffers. Gamma starts at 1 and the running variance starts at 1, so the first evaluation has a finite normalisation.

## FusedCBR2d::train

Turns training on for this layer and for the inner convolution. Otherwise Conv2d would stay in eval and would not keep the cache backward needs.

## FusedCBR2d::eval

Switches the layer and the convolution to inference. Batch-norm then reads the running stats, and the convolution itself uses the same math as in train.

## FusedCBR2d::configure_bn_descriptors

Rebuilds the NCHW and spatial BN descriptors only when the convolution output shape changes. Derive keeps the gamma layout consistent with CUDNN_BATCHNORM_SPATIAL.

## FusedCBR2d::apply_bn_leaky_into

In train, computes the batch moments, then in both modes finishes the statistics and launches the BN+LeakyReLU fusion. An empty tensor or a zero spatial plane returns early, because the reduction would divide by zero.

## FusedCBR2d::forward

Runs convolution-with-bias, then the BN and LeakyReLU fusion into the cache. In train it keeps the convolution output, because the batch-norm backward needs x from before normalisation.

## FusedCBR2d::backward

LeakyReLU derivative first, then cudnnBatchNormalizationBackward, then the convolution backward. The gradient that reaches the convolution is already past the activation and batch-norm, so Conv2d does not see the raw loss.

## FusedCBR2d::step

Copies the optimiser hyperparameters onto the inner convolution and takes an SGD step on gamma and beta. Without the learning_rate copy, the convolution step would keep the inner layer's previous step.

## FusedCBR2d::clip_gradients

Clips the convolution gradients and gamma and beta. A bound <= 0 is a no-op, as in the base clip_gradients.

## FusedCBR2d::get_parameters

Appends gamma, beta, and the running stats to the convolution parameter map. One network save covers both the convolution weights and the normalisation.

## FusedCBR2d::set_parameters

Loads the convolution parameters and the four batch-norm tensors. The copy checks the size so a weight file cannot change the channel count.

## FusedCBR2d::to

Forwards the device to the convolution and stays on the GPU. The batch-norm buffers have no host copy.
