# Losses

Loss types are not layers. They do not hold parameters and they do not implement `step`. Each one has a static `loss` and a static `loss_derivative`. Both run on GPU tensors and return GPU tensors. Reading the scalar requires `to_host` at the call site.

## `CrossEntropyLoss`

Mean softmax cross-entropy on logits `[N, C]` and a dense target of the same shape. The shipped loaders pass a one-hot target. Softmax is inside both `loss` and `loss_derivative`. The gradient is `softmax(logits) - target`, averaged over the batch. Callers pass logits.

## `MSELoss`

Mean squared error over all elements, and the matching derivative. It has no dataset-specific layout.
