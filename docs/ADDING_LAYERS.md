# Adding a generic layer

This tutorial adds a new **core** layer. Application networks (YOLO, SimpleCNN) should **compose** existing layers in `benchmarks/models/` instead of growing `DeepLearnLib`.

A layer is generic when it:

- inherits `Layer`
- speaks `dl::Tensor` on GPU
- does not encode a dataset or a full architecture

## 1. Header

Create `include/DeepLearnLib/MyLayer.hpp`:

```cpp
#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <map>
#include <optional>
#include <string>

/**
 * @brief Example elementwise scale layer (documentation template).
 */
class MyLayer : public Layer
{
public:
    explicit MyLayer(float scale);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative, cudaStream_t stream = 0)
        -> dl::Tensor override;
    void step(cudaStream_t stream = 0) override;
    auto get_parameters() -> std::map<std::string, dl::Tensor> override;
    void set_parameters(const std::map<std::string, dl::Tensor>& params) override;
    auto to(dl::Device device) -> void override;

private:
    float scale_;
    std::optional<dl::Tensor> output_cache_;
    std::optional<dl::Tensor> grad_input_cache_;
};
```

## 2. Implementation rules

Put the `.cpp` in `src/` and add it to `DEEPLEARN_SOURCES` in `src/CMakeLists.txt`. If the file contains `__global__` kernels, also list it in `DEEPLEARN_CUDA_SOURCES` so CMake compiles it as CUDA.

**Stay on GPU.** Do not `to_host()` in `forward`/`backward`.

**Reuse buffers.**

```cpp
auto MyLayer::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::StreamGuard stream_guard(stream);
    dl::Tensor& output = dl::Tensor::ensure(
        output_cache_, input_tensor.get_shape(), dl::Device::GPU, input_tensor.get_dtype());
    // write into output.data() / launch a kernel
    return output.as_view();
}
```

`ensure` allocates only when the shape changes. Returning `as_view()` avoids a D2D copy of the cache.

**In-place optimiser.** If the layer has weights, keep gradients as members and in `step()` call:

```cpp
weights_.sgd_update_(weights_gradient_, scaled_learning_rate(), /*decay=*/0.0F, parameter_clip_bound());
```

Do **not** allocate `weights_ = weights_ - lr * grad`.

**cuDNN / cuBLAS.** Convolution-like layers should use the C APIs (`cudnnConvolutionForward`, `cublasSgemm` via `matmul_into`) and wrap calls in `CHECK_CUDNN` / `CHECK_CUBLAS` / `CHECK_CUDA`.

**Error handling.** Throw `std::runtime_error` on shape mismatches; do not return half-updated GPU state.

## 3. Parameter I/O

`Network::save` / `load` round-trip `get_parameters()` / `set_parameters()`. Use stable names (`"weight"`, `"bias"`). Tensors in the map should be the live buffers (or copies with the same shape) so checkpoints restore training.

## 4. Tests

Add `tests/test_mylayer.cpp` and register it in `tests/CMakeLists.txt`. Cover:

- forward shape
- a tiny numeric backward check (finite difference or known closed form)
- `train()` vs `eval()` if behaviour differs

Run `./scripts/dev.sh` or `./build/tests/dllib_tests`.

## 5. Optional: fuse instead of adding a layer

If the new op always follows an existing one (e.g. bias + activation), prefer a fused kernel or a composite like `FusedCBR2d` rather than two global-memory passes. Document the fusion in [ARCHITECTURE.md](ARCHITECTURE.md).

## 6. Using the layer in a model

Do **not** add topologies to `src/`. In `benchmarks/models/` (or a new file there):

```cpp
layers_.push_back(std::make_shared<MyLayer>(0.5F));
layers_.push_back(std::make_shared<FullyConnected>(in, out));
```

Link the app with `DeepLearnLib` (and `DeepLearnModels` if you extend YOLO/SimpleCNN).
