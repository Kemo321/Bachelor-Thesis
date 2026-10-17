# Losses.cpp

Cross-entropy on the GPU: shape checks, row-wise softmax, and a reduction to a scalar.

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

## softmax_minus_target_kernel

`gradient[i] = (probability - target) * inv_batch`. This is the cross-entropy derivative with respect to the logits, already divided by the batch.

## softmax_probabilities

Allocates a tensor with the shape of the logits and launches `softmax_rows_kernel` with one block per 256 rows on `dl::current_stream()`. Checks `cudaGetLastError`.

## CrossEntropyLoss::loss

`require_same_gpu` and `require_rank2`. Softmax by rows, then `cross_entropy_rows_kernel`, both on `dl::current_stream()`. `sum().to_host(current_stream())` waits until that stream finishes, including a non-blocking image-training stream, and the total is divided by the batch. The result is a scalar on the GPU.

## CrossEntropyLoss::loss_derivative

The same shape checks. Softmax, then `softmax_minus_target_kernel` on the current stream with scale `1/batch`.
