# Softmax.cpp

Channel cuDNN softmax, with the output cached for the derivative.

## require_gpu

Rejects a host tensor or a null device pointer. cuDNN softmax reads the device memory named by the descriptor, with no host-to-device copy.

## Softmax::configure_descriptor

Sets the cuDNN NCHW descriptor for rank 2 or rank 4. Rank 2 is treated as [N, C, 1, 1], because channel softmax requires four dimensions.

## Softmax::forward

Computes accurate softmax in channel mode and keeps the output in the cache. cuDNN backward needs the softmax values, not the input alone, and ensure does not allocate when the shape is unchanged.

## Softmax::backward

Computes the softmax derivative from the cached output and the gradient. The cache flag is cleared afterwards so a stale y is not differentiated.
