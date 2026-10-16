# Gru.cpp

One-layer cuDNN GRU. The input is batch-major `[N, T, F]`. `forward` returns the final hidden state `[N, H]`. Math mode stays `CUDNN_DEFAULT_MATH` so the gate layout matches LibTorch.

When the hidden size is at most 128 and the device supports it, the layer uses `CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H`. That algorithm wants time-major data, so the forward and backward transpose. If the persistent weight layout does not match the expected input gates, the layer falls back to `CUDNN_RNN_ALGO_STANDARD` and keeps the batch-major buffers.

## batch_to_time_kernel

Copies `[N, T, F]` into `[T, N, F]`.

## time_to_batch_kernel

Copies `[T, N, F]` back to `[N, T, F]`.

## launch_transpose

Picks the kernel from `to_time_major` and launches it on the current stream.

## scatter_paired_kernel

Writes a dense host-filled matrix into the strided cuDNN weight or bias tensor.

## float_count_for_bytes

Rounds a byte size up to a number of floats so a `dl::Tensor` can own the buffer.

## require_contiguous_rank3

Rejects a tensor that is not a contiguous GPU rank-3 matrix. The RNN data descriptor assumes that layout.

## ensure_zero_like

Creates or resizes an Adam or SGD buffer to the weight shape and zeros it when the shape changes.

## copy_same_size

Copies a parameter when the element counts match.

## Gru::Impl

Owns the cuDNN descriptors, the packed weight space, and the workspace. Dropout probability is 0. The initial hidden state `hx` is cleared inside `prepare` when that buffer is allocated, not on every `forward`.

## Gru::Gru

Builds the implementation and marks the layer as GPU-only.

## Gru::~Gru

The destructor is out of line so `unique_ptr<Impl>` sees the complete type.

## Gru::input_size

Returns the feature count `F`.

## Gru::hidden_size

Returns the hidden count `H`.

## Gru::forward

Prepares descriptors for the batch, copies or transposes the input, and runs `cudnnRNNForward`. The returned view is the final hidden state.

## Gru::backward

Requires a training forward. The upstream gradient is `[N, H]` on the final state. Data and weight gradients come from the cuDNN v8 backward calls. A time-major layer transposes `dx` back to batch-major.

## Gru::step

Updates the packed weight space with Adam when `use_adam` is set, otherwise SGD with optional momentum. A frozen layer returns immediately.

## Gru::clip_gradients

Clips the packed weight gradient. A bound `<= 0` turns clipping off.

## Gru::get_parameters

Returns a view named `weight_space` of the packed cuDNN weights.

## Gru::set_parameters

Copies `weight_space` into the packed buffer when the sizes match.

## Gru::to

Keeps the parameters on the GPU. A CPU request throws.

## Gru::export_linear_layers

Reads the four gate matrices and biases out of the cuDNN weight space as dense host matrices. Tests use this to compare the layout with LibTorch.
