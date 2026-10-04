# Network.cpp

Builds a network from layers, runs a forward pass and a YOLO epoch, clips the gradient, and writes weights to a binary file.

## write_pod

Writes `sizeof(T)` bytes of the value to the stream. When the stream reports an error, throws with the given path in the message.

## read_pod

Reads `sizeof(T)` bytes into `value`. A short read or an I/O error throws: the weight file ended earlier than the layout requires.

## write_bytes

Writes a raw buffer. A length of zero is skipped so `write` is not called with an empty payload. A stream error throws.

## read_bytes

Reads a raw buffer. A length of zero is skipped. A short read throws.

## Network::Network

Takes ownership of the layer vector. A null in the vector throws. Writes `learning_rate` and `gradient_clip` into every layer.

## Network::set_gradient_clip

Stores the threshold in `gradient_clip_` and calls `sync_layer_optimizer_state`, so the layers receive the new value immediately.

## Network::sync_layer_optimizer_state

Copies `gradient_clip_` into the `gradient_clip` field of every layer.

## Network::gradient_clip

Returns `gradient_clip_`.

## Network::clip_loss_gradient

When the threshold is <= 0, returns `gradient.as_view()` with no copy. Otherwise takes `scaled_gradient_clip(gradient_clip_)`. `Tensor::ensure` keeps one buffer in `loss_grad_clip_cache_`, so a shape that does not change does not allocate GPU memory on every step. When the size is non-zero, copies the gradient device-to-device and calls `clamp_(-clip, clip)`. Returns a view of that buffer.

## Network::clip_parameter_gradients

When the threshold is <= 0, returns. Otherwise passes `scaled_gradient_clip` and the given stream to every layer through `clip_gradients`.

## Network::forward

`StreamGuard` sets the stream for the duration of the call. The input receives a `view` of its own shape, then each layer computes `forward` and the result goes through `view` again (the shape, without a copy of the data). Under `DEBUG_NUMERICS`, calls `assert_finite` after the layer. Returns the last tensor.

## Network::fit

A negative epoch count throws. In each epoch: `forward`, then when `verbose != 0`, every 10 epochs and on the last one, brings `YOLOLoss::loss` to the host (an empty tensor throws) and logs it. The gradient is `clip_loss_gradient` of `YOLOLoss::loss_derivative`. Layers run from the end through `backward`. At the end every layer receives `step`. The loss is not brought to the host when a log line is not needed.

## Network::save

Opens the file in binary mode. Writes the layer count as `int32`. For each layer: the parameter count from `get_parameters`, then for each parameter the name length, the name bytes, the rank, the dimensions as `int32`, and the fp32 weights from `to_host`. Flushes after the loop. An open or flush error throws. Logs the path at the end.

## Network::load

Opens the file. The layer count must equal `layers_.size()`, otherwise throws. For a layer, reads the parameter count. Each parameter: name length (negative throws), the name, the rank (negative throws), the dimensions, and as many floats as the product of the dimensions. Rank 0 sets the element count to 1. The tensor is created with `Tensor::from_host` on the GPU. The map goes to `set_parameters`. Logs the path at the end.
