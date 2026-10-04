# FullyConnected.cpp

Dense layer: a matrix product on the GPU, with gradient accumulation and an SGD step.

## UniformFill::operator()

Maps an index onto a value in [low, high) by hashing. Initialisation does not call a host generator and does not synchronise the device.

## uniform_fill_kernel

Applies UniformFill across the tensor elements. The range and seed travel in the functor, so the kernel does not read extra buffers.

## fill_constant_kernel

Writes a constant other than zero. cudaMemset can zero a buffer, not set another value.

## fill_uniform

Launches the uniform-initialisation kernel. An empty tensor is skipped so the launch does not use a grid of zero blocks.

## fill_constant

Zeros the buffer with cudaMemsetAsync, and writes any other constant with the kernel. Gradients start at zero and do not need the kernel.

## ensure_zero_like

Creates or resizes the velocity buffer and zeros it when the shape changes. SGD momentum needs a velocity with the weight's shape; a fresh allocation left unzeroed would be garbage.

## require_gpu

Rejects a host tensor or a null device pointer. The matrix product stays on the GPU and there is no host-to-device copy here.

## copy_same_size

Copies weights device-to-device, converting on the current stream when the dtype differs. A loader may supply a type other than the layer buffer's.

## require_rank2

Requires a [batch, features] matrix with the given column count. The feature dimension is the contract of the product with the weight [input, output].

## fullyconnected_weight_shape

Checks that the dimensions are positive and returns the weight shape [input, output]. The error must fire before the constructor allocates the tensor.

## FullyConnected::FullyConnected

Draws weights and bias from 1/sqrt(fan_in) and zeros the gradients. inertia_ is stored because backward uses it as the beta that accumulates dW, not as SGD momentum.

## FullyConnected::forward

Computes Y = X W + b and caches the input after a dtype conversion when needed. The output view comes from ensure, so the next forward overwrites the same buffer.

## FullyConnected::backward

Accumulates dW and db and computes dX = dY W^T. inertia_ accumulates the parameter gradients; dX is overwritten because the input-gradient cache is not summed across calls.

## FullyConnected::step

Updates the weights with SGD, and when momentum > 0 keeps velocity in an optional buffer. A frozen layer returns immediately and does not touch the weights.

## FullyConnected::clip_gradients

Clips the weight and bias gradients to a symmetric range. A bound <= 0 turns clipping off.

## FullyConnected::get_parameters

Returns views of the weights and bias for saving the network. Gradients and momentum velocity are not written to the file.

## FullyConnected::set_parameters

Loads the weights and bias with a device-to-device copy. The buffer shapes stay as they were at construction.

## FullyConnected::to

Leaves the parameters on the GPU. The matrix product has no host path.
