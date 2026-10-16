# Coverage

This report maps library sources, application models, and the shared benchmark headers onto the GoogleTest suite.
A file counts as covered when a test names it. Training and forward programs are drivers: they are not part of that percentage, because the loop, the CSV, and the models they call are what the tests check.

- Files covered: **55 / 72** (76.4%)
- Line coverage: not in this file. Run with `--lcov coverage.filtered.info` after a `--coverage` build. CI does that on the debug job.

## Library, models, and shared headers

| File | Tests | Lines |
| --- | --- | --- |
| `benchmarks/common/classification_eval.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `benchmarks/common/classification_vis.hpp` | — | — |
| `benchmarks/common/custom_step.hpp` | — | — |
| `benchmarks/common/experiment_config.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `benchmarks/common/forward_plan.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `benchmarks/common/image_train_custom.hpp` | — | — |
| `benchmarks/common/image_train_torch.hpp` | — | — |
| `benchmarks/common/indexed_train_custom.hpp` | — | — |
| `benchmarks/common/indexed_train_torch.hpp` | — | — |
| `benchmarks/common/prefetch_batch.hpp` | — | — |
| `benchmarks/common/row_batch.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `benchmarks/common/run_metrics.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `benchmarks/common/split_metrics.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `benchmarks/common/tabular_common.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `benchmarks/common/tabular_pipeline.hpp` | — | — |
| `benchmarks/common/torch_optim.hpp` | — | — |
| `benchmarks/common/torch_paired_init.hpp` | — | — |
| `benchmarks/common/train_loop.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `benchmarks/models/cnn_shapes.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `benchmarks/models/GruClassifier.cpp` | `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp` | — |
| `benchmarks/models/GruClassifier.hpp` | `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp` | — |
| `benchmarks/models/SimpleCNN.cpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp` | — |
| `benchmarks/models/SimpleCNN.hpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp` | — |
| `benchmarks/models/TabularMLP.cpp` | `tests/models/test_model_parity.cpp`, `tests/models/test_tabular_mlp.cpp` | — |
| `benchmarks/models/TabularMLP.hpp` | `tests/models/test_model_parity.cpp`, `tests/models/test_tabular_mlp.cpp` | — |
| `benchmarks/models/torch/TorchGruClassifier.hpp` | `tests/models/test_model_parity.cpp` | — |
| `benchmarks/models/torch/TorchSimpleCNN.hpp` | `tests/models/test_model_parity.cpp` | — |
| `benchmarks/models/torch/TorchTabularMLP.hpp` | `tests/models/test_model_parity.cpp` | — |
| `include/DeepLearnLib/core/Batch.hpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_batchnorm2d.cpp`, `tests/layers/test_flatten.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/loaders/test_classification_loader.cpp`, `tests/loaders/test_packed_image_loader.cpp`, `tests/losses/test_generic.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp`, `tests/models/test_tabular_mlp.cpp` | — |
| `include/DeepLearnLib/core/Layer.hpp` | `tests/core/test_network.cpp`, `tests/layers/test_activations.cpp`, `tests/layers/test_batchnorm2d.cpp`, `tests/layers/test_conv2d.cpp`, `tests/layers/test_flatten.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_gru.cpp`, `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/losses/test_generic.cpp`, `tests/losses/test_numerics.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp`, `tests/models/test_tabular_mlp.cpp` | — |
| `include/DeepLearnLib/core/Network.hpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/losses/test_numerics.cpp` | — |
| `include/DeepLearnLib/core/Precision.hpp` | — | — |
| `include/DeepLearnLib/core/Tensor.hpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_activations.cpp`, `tests/layers/test_batchnorm2d.cpp`, `tests/layers/test_conv2d.cpp`, `tests/layers/test_flatten.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_gru.cpp`, `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/loaders/test_classification_loader.cpp`, `tests/loaders/test_packed_image_loader.cpp`, `tests/losses/test_generic.cpp`, `tests/losses/test_numerics.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp`, `tests/models/test_tabular_mlp.cpp`, `tests/test_helpers.hpp` | — |
| `include/DeepLearnLib/layers/BatchNorm2d.hpp` | `tests/layers/test_batchnorm2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | — |
| `include/DeepLearnLib/layers/Conv2d.hpp` | `tests/layers/test_conv2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | — |
| `include/DeepLearnLib/layers/Dropout.hpp` | `tests/layers/test_activations.cpp` | — |
| `include/DeepLearnLib/layers/Flatten.hpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_flatten.cpp` | — |
| `include/DeepLearnLib/layers/FullyConnected.hpp` | `tests/core/test_network.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/losses/test_numerics.cpp` | — |
| `include/DeepLearnLib/layers/Gru.hpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/layers/test_gru.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp` | — |
| `include/DeepLearnLib/layers/LeakyReLU.hpp` | `tests/layers/test_activations.cpp` | — |
| `include/DeepLearnLib/layers/MaxPool2d.hpp` | `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | — |
| `include/DeepLearnLib/layers/Softmax.hpp` | `tests/losses/test_generic.cpp` | — |
| `include/DeepLearnLib/loaders/ClassificationLoader.hpp` | `tests/loaders/test_classification_loader.cpp` | — |
| `include/DeepLearnLib/loaders/CSVLoader.hpp` | `tests/losses/test_generic.cpp` | — |
| `include/DeepLearnLib/loaders/Ecg200Loader.hpp` | `tests/layers/test_gru.cpp` | — |
| `include/DeepLearnLib/loaders/PackedImageLoader.hpp` | `tests/loaders/test_packed_image_loader.cpp` | — |
| `include/DeepLearnLib/losses/Losses.hpp` | `tests/losses/test_generic.cpp`, `tests/losses/test_numerics.cpp` | — |
| `include/DeepLearnLib/utils/ExperimentSeed.hpp` | `tests/benchmarks/test_train_metrics.cpp` | — |
| `include/DeepLearnLib/utils/Logger.hpp` | — | — |
| `include/DeepLearnLib/utils/Nvtx.hpp` | — | — |
| `include/DeepLearnLib/utils/ParallelFor.hpp` | `tests/loaders/test_classification_loader.cpp` | — |
| `include/DeepLearnLib/utils/Profiler.hpp` | — | — |
| `include/DeepLearnLib/utils/SafeMath.hpp` | `tests/losses/test_numerics.cpp` | — |
| `src/core/Network.cpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/losses/test_numerics.cpp` | — |
| `src/core/Precision.cpp` | — | — |
| `src/core/Tensor.cpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_activations.cpp`, `tests/layers/test_batchnorm2d.cpp`, `tests/layers/test_conv2d.cpp`, `tests/layers/test_flatten.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_gru.cpp`, `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/loaders/test_classification_loader.cpp`, `tests/loaders/test_packed_image_loader.cpp`, `tests/losses/test_generic.cpp`, `tests/losses/test_numerics.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp`, `tests/models/test_tabular_mlp.cpp`, `tests/test_helpers.hpp` | — |
| `src/layers/BatchNorm2d.cpp` | `tests/layers/test_batchnorm2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | — |
| `src/layers/Conv2d.cpp` | `tests/layers/test_conv2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | — |
| `src/layers/Dropout.cpp` | `tests/layers/test_activations.cpp` | — |
| `src/layers/Flatten.cpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_flatten.cpp` | — |
| `src/layers/FullyConnected.cpp` | `tests/core/test_network.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/losses/test_numerics.cpp` | — |
| `src/layers/Gru.cpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/layers/test_gru.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp` | — |
| `src/layers/LeakyReLU.cpp` | `tests/layers/test_activations.cpp` | — |
| `src/layers/MaxPool2d.cpp` | `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | — |
| `src/layers/Softmax.cpp` | `tests/losses/test_generic.cpp` | — |
| `src/loaders/ClassificationLoader.cpp` | `tests/loaders/test_classification_loader.cpp` | — |
| `src/loaders/CSVLoader.cpp` | `tests/losses/test_generic.cpp` | — |
| `src/loaders/Ecg200Loader.cpp` | `tests/layers/test_gru.cpp` | — |
| `src/loaders/PackedImageLoader.cpp` | `tests/loaders/test_packed_image_loader.cpp` | — |
| `src/losses/Losses.cpp` | `tests/losses/test_generic.cpp`, `tests/losses/test_numerics.cpp` | — |
| `src/utils/Logger.cpp` | — | — |
| `src/utils/Profiler.cpp` | — | — |

## Files with no matching test

- `benchmarks/common/classification_vis.hpp`
- `benchmarks/common/custom_step.hpp`
- `benchmarks/common/image_train_custom.hpp`
- `benchmarks/common/image_train_torch.hpp`
- `benchmarks/common/indexed_train_custom.hpp`
- `benchmarks/common/indexed_train_torch.hpp`
- `benchmarks/common/prefetch_batch.hpp`
- `benchmarks/common/tabular_pipeline.hpp`
- `benchmarks/common/torch_optim.hpp`
- `benchmarks/common/torch_paired_init.hpp`
- `include/DeepLearnLib/core/Precision.hpp`
- `include/DeepLearnLib/utils/Logger.hpp`
- `include/DeepLearnLib/utils/Nvtx.hpp`
- `include/DeepLearnLib/utils/Profiler.hpp`
- `src/core/Precision.cpp`
- `src/utils/Logger.cpp`
- `src/utils/Profiler.cpp`

## Training and benchmark drivers

These programs are the paired custom and Torch entry points. They are compiled by the release job and are not unit tests.

- `benchmarks/forward/bench_forward_custom.cpp` — paired binary
- `benchmarks/forward/bench_forward_torch.cpp` — paired binary
- `benchmarks/micro/bench_micro_common.hpp` — paired binary
- `benchmarks/micro/bench_micro_more.cpp` — paired binary
- `benchmarks/micro/bench_micro_ops.cpp` — paired binary
- `benchmarks/train/train_cifar_custom.cpp` — paired binary
- `benchmarks/train/train_cifar_torch.cpp` — paired binary
- `benchmarks/train/train_gru_custom.cpp` — paired binary
- `benchmarks/train/train_gru_torch.cpp` — paired binary
- `benchmarks/train/train_mnist_custom.cpp` — paired binary
- `benchmarks/train/train_mnist_torch.cpp` — paired binary
- `benchmarks/train/train_tabular_custom.cpp` — paired binary
- `benchmarks/train/train_tabular_torch.cpp` — paired binary
