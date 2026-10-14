# Flatten.cpp

Flattens a tensor to a [batch, rest] matrix by view, without copying.

## require_gpu

Rejects a host tensor or a null device pointer. The flatten is only a view and does not copy data on the GPU.

## Flatten::forward

Changes the view to [batch, rest] without copying data. The input shape stays in the cache because backward has to restore it.

## Flatten::backward

Restores the gradient to the shape from before the flatten. This is only a view: the layer has no weights and does not change the values.
