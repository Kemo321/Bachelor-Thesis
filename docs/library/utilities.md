# Utilities

These types sit next to layers. They are not a training program. How applications loop over epochs is [Usage](../usage/training.md).

## `Network`

`Network` holds `vector<shared_ptr<Layer>>` in forward order. The constructor copies `learning_rate` and `gradient_clip` onto every layer.

| Method | Behaviour |
| --- | --- |
| `forward` | Walks layers head to tail. Each output is re-wrapped with `view`. |
| `clip_loss_gradient` | Clamps `dL/dpred` into a cached buffer when `gradient_clip > 0`. Returns a view of the input when clipping is off. |
| `clip_parameter_gradients` | Calls `Layer::clip_gradients` on every layer. |
| `save` / `load` | Binary file of each layer's `get_parameters()` map. The format is private to this method. A `.pt` suffix on a path does not make the file a PyTorch state dict. |
| `fit` | Repeats `forward`, `YOLOLoss`, reverse `backward`, and `step` on tensors that are already on the device. The loss is fixed as `YOLOLoss`. Applications do not call `fit`; they own the loop so they can schedule the learning rate and write metrics. |

`set_gradient_clip` updates the bound and pushes it to the layers.

## Precision

`dl::set_mixed_precision(enabled, loss_scale)` and `dl::configure_precision` select the process compute dtype. The default is FP32. FP16 uses the loss scale (default `1024`) so small gradients survive the half exponent. Call this before constructing layers. Weights are allocated in the compute dtype.

`dl::MixedPrecisionGuard` restores the previous policy when it leaves scope.

`dl::scaled_gradient_clip` multiplies a clip bound by the current scale. `Layer::scaled_learning_rate` divides the learning rate by that scale.

## `Profiler`

`start` records a CUDA event on the current stream and does not synchronise. `stop` synchronises and returns milliseconds. `get_vram_usage_mb` reads device memory for the process from the driver.

`dl::NvtxRange` pushes an NVTX range for Nsight. It does not change results.

## mAP and box helpers

`Detection` is an axis-aligned box: `x`, `y`, `width`, `height`, `score`, `class_id`.

`mean_average_precision` is VOC 11-point average precision at a fixed IoU (default `0.5`). Predictions are matched per class, highest score first. Each ground-truth box is used at most once. The return value is the unweighted mean of per-class AP. The function runs on the host, on `vector<Detection>`.

`utils.hpp` (requires OpenCV) adds `calculate_iou` for `cv::Rect`, `apply_nms`, `decode_yolo_tensor`, and `draw_detections`. Decoding turns a flat YOLO grid into boxes. NMS and drawing are host-side.

## Logging

`dl::Logger` is a process-wide spdlog logger. `LOG_INFO`, `LOG_DEBUG`, `LOG_ERROR`, and `LOG_FLUSH` are the macros used by binaries. Logging is the caller's choice. The tensor and layer implementations do not log on the hot path except through `CHECK_*` when a CUDA call fails.
