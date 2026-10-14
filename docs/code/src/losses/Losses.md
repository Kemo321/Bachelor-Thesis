# Losses.cpp

MSE and cross-entropy on the GPU: shape checks, row-wise softmax, and reductions to a scalar.

## require_same_gpu

Throws when the target or the prediction is not on the GPU, the shapes or the sizes differ, or, at a non-zero size, either pointer is null. `name` is included in the message.

## require_rank2

Requires rank 2 and positive `[batch, classes]` dimensions. Softmax in this file runs by rows, so another layout does not apply.

## scalar_from_host

Builds a tensor `{1}` on the GPU from one value. The loss comes back as a tensor, like the rest of the graph.

## softmax_rows_kernel

One thread per row. Subtracts the row maximum before `exp`, so the result stays in `(0, 1]` and does not overflow to inf. When the maximum is not finite, the row receives the uniform distribution `1/classes`. Otherwise divides `exp` by the sum, floored at `kSafeEps`.

## cross_entropy_rows_kernel

One thread per row. Computes the sum of `-target * log(p)` over the classes. `clamp_unit` clamps `p` to `[eps, 1]` before the log. The result is stored in `row_loss[row]`.

## mse_sqdiff_sum_kernel

Sums `(prediction - target)^2` over the elements. A thread gathers its slice, the block reduces it in shared memory (`kReduceThreads` = 256), and thread 0 does `atomicAdd` into `out`.

## mse_grad_kernel

`gradient[i] = (prediction - target) * scale`. The host computes the `2/N` scale in `MSELoss::loss_derivative`.

## softmax_minus_target_kernel

`gradient[i] = (probability - target) * inv_batch`. This is the cross-entropy derivative with respect to the logits, already divided by the batch.

## softmax_probabilities

Allocates a tensor with the shape of the logits and launches `softmax_rows_kernel` with one block per 256 rows. The kernel runs on the default stream (the launch does not pass a stream). Checks `cudaGetLastError`.

## MSELoss::loss

`require_same_gpu`. Size 0 returns the scalar 0 without a kernel. Otherwise zeroes a scalar on the current stream, sums the squared differences (`mse_sqdiff_sum_kernel`, at most 1024 blocks), and divides the sum by the element count. `to_host` synchronizes the stream, because the mean is a number on the CPU; the scalar goes back to the GPU through `scalar_from_host`.

## MSELoss::loss_derivative

`require_same_gpu`. Allocates a gradient with the shape of the prediction. Size 0 returns it immediately. Otherwise the scale is `2 / N`, and `mse_grad_kernel` on the current stream fills the buffer.

## CrossEntropyLoss::loss

`require_same_gpu` and `require_rank2`. Softmax by rows, then `cross_entropy_rows_kernel` (default stream). The sum of the row losses comes to the host through `sum().to_host()` — that waits on the stream — and is divided by the batch. The result is a scalar on the GPU.

## CrossEntropyLoss::loss_derivative

The same shape checks. Softmax, then `softmax_minus_target_kernel` on the current stream with scale `1/batch`.
