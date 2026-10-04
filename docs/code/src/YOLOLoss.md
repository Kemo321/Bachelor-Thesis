# YOLOLoss.cpp

YOLOv1 loss on a 7×7 grid: box IoU, the mean cell loss, and its gradient with respect to the prediction.

Cell layout: two boxes of `x, y, w, h, confidence` (offsets 0–4 and 5–9), then `num_classes` classes from offset 10. In the target, offset 4 is the object mask, not the confidence.

## ceil_div

`(value + divisor - 1) / divisor`. Runs on the host and on the device, with no library call.

## clampf

Returns `value` when it is not less than `lo`, otherwise `lo`. An intersection width must not be negative.

## box_iou

IoU of two boxes given as center and size. Computes the rectangles, floors the intersection width and height at zero, floors the areas at `kSafeEps`, and divides through `safe_div`.

## decode_center

`(raw + grid_index) / 7`. An offset inside the cell becomes the center coordinate on the grid.

## cell_base

Linear index of the start of a cell: `(((batch * 7) + row) * 7 + col) * final_dim`.

## yolo_iou_kernel

One thread per pair. Reads 4 floats from `box1` and `box2` and writes `box_iou` into `iou[idx]`.

## yolo_loss_forward_kernel

One thread per cell (`batch * 49`). IoU of both boxes against the target is computed after `decode_center` on the centers. The responsible box is the one with the higher IoU; a tie stays with the first (`iou2 > iou1`). The object mask multiplies the responsibility.

The coordinate loss is `kLambdaCoord` (5) times the sum of the squared `x, y` differences and the squared differences of `sqrt(w)` and `sqrt(h)`, only for the responsible box. Confidence: the square of `(confidence - IoU)` for the responsible box, and `kLambdaNoobj` (0.5) times the square of the confidence for a box that is not responsible. Classes: the sum of squared differences from offset 10, times the object mask. `cell_loss` receives the sum of the three terms.

## yolo_loss_backward_kernel

The same choice of responsible box as the forward pass. Center gradient: `2 * kLambdaCoord * (prediction - target) * responsibility * inv_batch`. Width and height gradient: `kLambdaCoord * (sqrt(p) - sqrt(t)) / sqrt(p)`, zeroed when the side is not greater than `kSafeEps`, times responsibility and `inv_batch`. Confidence: `(2 * (c - IoU) * responsibility + 2 * kLambdaNoobj * c * not_responsible) * inv_batch`. Class: `2 * difference * object_mask * inv_batch`.

## yolo_mean_loss_kernel

One block. Threads sum `cell_loss` in a loop, reduce in shared memory, and thread 0 writes `sum * inv_batch` to `mean_loss[0]`.

## launch_config

Block count at 256 threads, at least one, including when `count` is zero. `ceil_div` rounds up.

## require_gpu_pair

Throws when the target or the prediction is not on the GPU, or a pointer is null. It does not check the shape here — `as_yolo_grid` does that.

## as_yolo_grid

Rank 4 must be `[batch, 7, 7, final_dim]` and comes back as a `view` of that shape. Rank 2 must have second dimension `49 * final_dim` and comes back as a view of `[batch, 7, 7, final_dim]`. Another rank throws. The data is not copied.

## yolo_workspace

Returns a static `YoloWorkspace` with three `optional<Tensor>` slots: `cell_loss`, `grad`, `scalar`. The grid size repeats every step, so `Tensor::ensure` does not allocate them again while the shape matches.

## YOLOLoss::calculate_iou

`require_gpu_pair`. The sizes must be equal and divisible by 4, otherwise throws. The box count is `size / 4`. An empty input returns an empty vector without a kernel. Otherwise `yolo_iou_kernel` on the current stream.

## YOLOLoss::loss

NVTX range `YOLOLoss_Loss` and a `StreamGuard` on the given stream. `num_classes` must be positive. Both tensors go through `as_yolo_grid` with `final_dim = 10 + num_classes`. A different batch throws. `ensure` keeps `cell_loss` of length `batch * 49` and the mean scalar, so `cudaMalloc` is not called on every call. Forward kernel, then `yolo_mean_loss_kernel` with `1/batch`. Returns `as_view()` of the scalar.

## YOLOLoss::loss_derivative

NVTX range `YOLOLoss_LossDerivative` and the same stream guard. The same checks as `loss`. Remembers whether the prediction arrived as rank 2. `ensure` keeps a gradient with the grid shape. The backward kernel receives `1/batch`. When the input was flat, returns a `view` of `[batch, 49 * final_dim]`. Otherwise `as_view()` of the grid.
