# LeakyReLU.cpp

Elementwise LeakyReLU on the GPU, with the input cached for the derivative.

## load_act

Loads an activation element as float. The template keeps one kernel per storage type, while the arithmetic still runs in float.

## store_act

Stores a float result into the activation buffer. A separate function so forward and backward do not repeat the cast on the store.

## leaky_forward_kernel

Computes LeakyReLU element by element. Positive values pass through unchanged and negative values are multiplied by the slope, with no second pass over memory.

## leaky_backward_kernel

Multiplies the output gradient by 1 or by the slope, according to the sign of the input. The derivative is constant on each half-axis, so the input saved by forward is enough.

## elementwise_grid

Computes the block grid of the elementwise kernel. Rounding up by the element count and kThreads covers the last, partial block.

## require_gpu

Rejects a host tensor or a null device pointer. The layer does not copy data onto the GPU inside forward.

## LeakyReLU::LeakyReLU

Stores the slope of the negative half-axis and sets the device to GPU. The slope is fixed; it is not a trained parameter.

## LeakyReLU::forward

Launches the LeakyReLU kernel and returns a view of the output cache. The input stays in the cache because backward needs its sign, and ensure does not allocate when the shape is unchanged.

## LeakyReLU::backward

Passes the gradient through the LeakyReLU derivative using the cached input. The cache flag is cleared so a later backward without a forward cannot reuse a stale sign.
