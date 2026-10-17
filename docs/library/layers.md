# Layers

`Layer` is the extension point. A subclass implements:

- `forward(x)` returns `y`. The tensor may alias an internal cache. It is invalid after another `forward` of the same layer.
- `backward(dy)` returns `dx`. Parameterised layers also write `dW` and `db` into members.
- `step()` reads those gradients and updates parameters. Layers without parameters leave it empty.
- `train()` / `eval()` select the cuDNN GRU mode: training or inference.
- `get_parameters()` / `set_parameters()` expose named tensors for `Network::save` and `Network::load`.

Optimiser fields live on the layer: `learning_rate`, `momentum`, `weight_decay`, `gradient_clip`. `step_learning_rate()` returns `learning_rate`. `parameter_clip_bound()` returns `gradient_clip` when clipping is on.

Weight decay is applied inside `sgd_update_` / `sgd_momentum_update_`, not inside `backward`. `backward` is only the derivative.

With `momentum == 0` the update is vanilla SGD. `gradient_clip == 0` disables clipping. A positive clip is an absolute bound on the value that enters the update.

## Backends

| Class | Implementation | Parameters | `train` vs `eval` |
| --- | --- | --- | --- |
| `Conv2d` | cuDNN convolution | weight, bias | same math |
| `MaxPool2d` | cuDNN max pooling | none | same math |
| `FullyConnected` | small kernel or cuBLASLt, else `matmul_into` | weight `[F_in, F_out]`, bias `[1, F_out]` | same math |
| `LeakyReLU` | device kernel | none | same math |
| `Flatten` | `Tensor::view` | none | same |
| `Softmax` | cuDNN accurate softmax, channel mode | none | same math |

## `Conv2d`

NCHW in, NCHW out. Kernel, stride, and padding are fixed at construction. Tensor, filter, and convolution descriptors are rebuilt when the input shape changes. Forward and backward use the cuDNN convolution API. The workspace size reported by cuDNN is stored with `ensure`.

Bias is a separate tensor. `Conv2d` adds it with cuDNN's tensor add. The activation descriptor is identity, so the add stays linear.

## `MaxPool2d`

cuDNN backward needs the forward input and the forward output, so both are cached. Window and stride are constructor arguments.

## `FullyConnected`

```text
Y = X W + b
X is [N, F_in], W is [F_in, F_out], b is [1, F_out]
```

A narrow product (`batch * F_in * F_out` at most 524288) is one kernel that writes `X W + b`. Wider products use a cached cuBLASLt GEMM with a bias epilogue. If that path is unavailable, forward falls back to `matmul_into` and `add_row_(b)`.

Backward uses logical transposes:

- `dW = X^T dY` with `transpose_a`;
- `db` is the row sum of `dY`;
- `dX = dY W^T` with `transpose_b`.

The third constructor argument, `inertia`, is the GEMM `beta` used when writing `dW` and `db`. `0` replaces the gradient buffer. It is not SGD momentum. SGD momentum is `Layer::momentum`.

`step` calls `sgd_update_` or, when momentum is non-zero, `sgd_momentum_update_`. The velocity buffer is allocated on the first momentum step and then reused.

## `LeakyReLU`

`y = x` when `x >= 0`, otherwise `slope * x`. The default slope is `0.1`. Forward caches the input so backward can scale the negative side.

## `Flatten`

View from `[N, ...]` to `[N, F]` and back. No kernel and no parameters.

## `Softmax`

`CUDNN_SOFTMAX_ACCURATE` subtracts the max before the exponential. `CUDNN_SOFTMAX_MODE_CHANNEL` normalises across the channel dimension.

`CrossEntropyLoss` already applies softmax inside the loss and the gradient. A training step that uses that loss passes logits and does not backpropagate through this layer.
