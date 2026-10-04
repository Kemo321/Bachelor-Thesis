# Conventions

The tree follows one style in the code that is exercised every day. A handful of older spellings remain. This page lists the rule and every break that is still in the headers, so a new file can match the majority.

## Rules that hold

| Kind | Spelling | Examples |
| --- | --- | --- |
| Types | `PascalCase` | `Conv2d`, `FusedCBR2d`, `CrossEntropyLoss`, `Tensor` |
| Functions and methods | `snake_case` | `split_dataset`, `get_shape`, `matmul_into`, `mean_average_precision` |
| In-place tensor methods | `snake_case` plus a trailing underscore | `add_`, `clamp_`, `sgd_update_` |
| Private data members | trailing underscore | `weights_`, `is_training_`, `output_cache_` |
| Public data members | no underscore | `Layer::learning_rate`, `Detection::class_id` |
| Constants | `k` prefix, camel tail | `dl::kSafeEps`, `dl::kDefaultGradientClip` |
| Local variables and parameters | `snake_case` | `batch_size`, `num_classes` |
| Macros | `SCREAMING_SNAKE` | `CHECK_CUDA`, `LOG_INFO` |
| Layer headers | file name equals the type | `FullyConnected.hpp` |
| Training binaries | `{role}_{dataset}_{stack}` | `train_voc_custom`, `inference_bccd_torch` |
| JSON keys | `snake_case` | `learning_rate`, `batch_size`, `voc_custom` |
| Return type | trailing `auto name() -> T` | almost every method in `include/DeepLearnLib/` |

Constructor parameters that would shadow a member take a `_val` suffix (`inertia_val`, `stride_val`, `slope_val`). The member itself has no suffix beyond the private underscore.

SGD momentum is `Layer::momentum`. Batch-norm's running-stat blend is a different number. `BatchNorm2d` stores it as `momentum_bn_`. `FusedCBR2d` stores the same concept as `bn_momentum_`. New code should use `momentum_bn_`.

`FullyConnected`'s `inertia` argument is GEMM beta for the gradient write. It is not `Layer::momentum`.

## Namespace

`dl` contains storage and process-wide CUDA state: `Tensor`, `Device`, `Dtype`, precision, logger, `SafeMath`, `parallel_for`, NVTX, cuBLAS/cuDNN handles, and the cuDNN descriptor wrappers.

`Layer`, every layer subclass, `Network`, the three loss types, the four loaders, `Profiler`, `Detection`, and the functions in `utils.hpp` and `mAP.hpp` are in the global namespace.

A new layer stays global, next to `Conv2d`. A new helper that only touches CUDA lifetime or numeric policy goes in `dl`. Do not invent a third namespace.

## Breaks

These are real. New parameters stay in `snake_case`.

| Location | What differs |
| --- | --- |
| `utils.hpp` | `calculate_iou`, `apply_nms`, `decode_yolo_tensor`, and `draw_detections` use a classic return type (`float calculate_iou(...)`) instead of a trailing return type. Inside `decode_yolo_tensor` the grid extent is `GRID_SIZE`. The loss kernel spells the same constant `kGridSize`. |
| `Tensor` | Metadata uses a `get_` prefix (`get_shape`, `get_device`, `get_dtype`). The pointer is both `data()` and `get_data()` (`get_data` is const-only). New code should call `data()`. |
| Header file names | Layer types match the file. Multi-symbol headers do not follow one pattern: `dataset.hpp` and `utils.hpp` are lowercase, `mAP.hpp` camel-cases the metric, `Losses.hpp` is plural because it holds `MSELoss` and `CrossEntropyLoss`. `ParallelFor.hpp`, `Precision.hpp`, `SafeMath.hpp`, `Nvtx.hpp`, and `Logger.hpp` are `PascalCase` even though they are not a single class of that name. |
| CMake project | `project(MiniC_DL ...)`. The target, the include directory, and the docs say `DeepLearnLib`. |
| Container and image | `yolo_dev_container`, image `yolo-bachelor-thesis`. The library name does not appear. |
| `CustomDataLoader` | The type name means "loader for the detection datasets", not "the Custom training stack". Classification has its own types. |

Application names are uniform and are not part of the list above: `train_`, `inference_`, `bench_`, `short_`, `overfit_`, then the dataset, then `_custom` or `_torch`. Pipeline keys in `config/experiments.json` match that (`voc_custom`, `cifar10_classification`, `tabular_iris`).

## What to match

New code uses `split_dataset`, `momentum_bn_`, `data()`, and a trailing return type. Parameters stay in `snake_case`. The header filenames in the breaks table stay as they are.
