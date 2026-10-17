# Conventions

The tree follows one style in the code that is exercised every day. A handful of older spellings remain. This page lists the rule and every break that is still in the headers, so a new file can match the majority.

Headers and sources use the same folders: `core/` (tensor, layer contract, network, batch), `layers/`, `losses/`, `loaders/`, and `utils/`. A new layer goes in `layers/` on both sides. Tests follow the same names, plus `tests/models/` for `SimpleCNN`.

## Rules that hold

| Kind | Spelling | Examples |
| --- | --- | --- |
| Types | `PascalCase` | `Conv2d`, `CrossEntropyLoss`, `Tensor` |
| Functions and methods | `snake_case` | `get_shape`, `matmul_into`, `to_host` |
| In-place tensor methods | `snake_case` plus a trailing underscore | `add_`, `clamp_`, `sgd_update_` |
| Private data members | trailing underscore | `weights_`, `is_training_`, `output_cache_` |
| Public data members | no underscore | `Layer::learning_rate`, `Layer::use_adam` |
| Constants | `k` prefix, camel tail | `dl::kSafeEps`, `dl::kDefaultGradientClip` |
| Local variables and parameters | `snake_case` | `batch_size`, `num_classes` |
| Macros | `SCREAMING_SNAKE` | `CHECK_CUDA`, `CHECK_CUDNN` |
| Layer headers | file name equals the type, under `layers/` | `layers/FullyConnected.hpp` |
| Training binaries | `{role}_{dataset}_{stack}` | `train_cifar_custom`, `train_gru_torch` |
| JSON keys | `snake_case` | `learning_rate`, `batch_size`, `ecg200` |
| Return type | trailing `auto name() -> T` | almost every method in `include/DeepLearnLib/` |

Constructor parameters that would shadow a member take a `_val` suffix (`inertia_val`, `stride_val`, `slope_val`). The member itself has no suffix beyond the private underscore.

SGD momentum is `Layer::momentum`.

`FullyConnected`'s `inertia` argument is GEMM beta for the gradient write. It is not `Layer::momentum`.

## Namespace

`dl` contains storage and process-wide CUDA state: `Tensor`, `Device`, `Dtype`, precision, logger, `SafeMath`, `parallel_for`, NVTX, cuBLAS/cuDNN handles, and the cuDNN descriptor wrappers.

`Layer`, every layer subclass, `Network`, `CrossEntropyLoss`, the loaders, and `Profiler` are in the global namespace.

A new layer stays global, next to `Conv2d`. A new helper that only touches CUDA lifetime or numeric policy goes in `dl`. Do not invent a third namespace.

## Breaks

These are real. New parameters stay in `snake_case`.

| Location | What differs |
| --- | --- |
| `Tensor` | Metadata uses a `get_` prefix (`get_shape`, `get_device`, `get_dtype`). The pointer is both `data()` and `get_data()` (`get_data` is const-only). New code should call `data()`. |
| Header file names | Layer types match the file. `Losses.hpp` is plural because it holds the loss helpers and `CrossEntropyLoss`. `ParallelFor.hpp`, `Precision.hpp`, `SafeMath.hpp`, `Nvtx.hpp`, and `Logger.hpp` are `PascalCase` even though they are not a single class of that name. |
| CMake project | `project(MiniC_DL ...)`. The target, the include directory, and the docs say `DeepLearnLib`. |

Application names are `train_` or `bench_`, then the dataset, then `_custom` or `_torch`. Pipeline keys in `config/experiments.json` use the dataset name (`cifar10_classification`, `tabular_iris`, `ecg200`).

## What to match

New code uses `momentum_bn_`, `data()`, and a trailing return type. Parameters stay in `snake_case`.
