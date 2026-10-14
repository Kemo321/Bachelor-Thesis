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

`set_gradient_clip` updates the bound and pushes it to the layers.

## Storage

Device tensors are FP32. `Layer::step_learning_rate` returns `learning_rate`. `parameter_clip_bound` returns `gradient_clip` when that field is positive, and `0` otherwise.

## `Profiler`

`start` records a CUDA event on the current stream and does not synchronise. `stop` synchronises and returns milliseconds. `get_vram_usage_mb` reads device memory for the process from the driver.

`dl::NvtxRange` pushes an NVTX range for Nsight. It does not change results.

## Logging

`dl::Logger` is a process-wide spdlog logger. `LOG_INFO`, `LOG_DEBUG`, `LOG_ERROR`, and `LOG_FLUSH` are the macros used inside the library. `log_to_file` can add another info-level file. Training programs do not call it and do not write a text log beside the metrics CSV. The tensor and layer implementations do not log on the hot path except through `CHECK_*` when a CUDA call fails.
