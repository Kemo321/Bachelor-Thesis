# Dropout.cpp

Bernoulli dropout on the GPU: a mask in train, the identity in eval.

## BernoulliMask::operator()

Draws a Bernoulli threshold from a hash of the index and the seed. A kept element stores the inverted-dropout scale, a dropped element stores zero, with no cuRAND state on the device.

## load_act

Loads an activation element as float. The same load serves the input and gradient dtypes, and the multiply by the mask is in float anyway.

## store_act

Stores a float result into the activation buffer. Forward and backward share one store so the cast is not duplicated.

## dropout_mask_kernel

Fills the dropout mask on the GPU. The mask is a separate buffer because the same pattern has to reach backward.

## dropout_apply_kernel

Multiplies a tensor by the mask elementwise. The same kernel serves forward and backward, because both are that multiply.

## elementwise_grid

Computes the block grid of the elementwise kernel. Rounding up by the element count and kThreads covers the last, partial block.

## require_gpu

Rejects a host tensor or a null device pointer. The mask is created on the GPU and there is no host-to-device copy here.

## Dropout::Dropout

Checks that the drop probability lies in [0, 1), and sets the mask seed. A value of 1 would zero the whole tensor, and the scale 1/(1-p) would be zero.

## Dropout::forward

In train, draws a Bernoulli mask and applies it to the input; in eval, returns a view of the input. The inverted-dropout scale lives in the mask, so inference does not multiply separately.

## Dropout::backward

In train, multiplies the gradient by the same mask as forward. In eval, or without a mask, the gradient is returned unchanged, because forward zeroed nothing.
