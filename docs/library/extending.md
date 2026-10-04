# Extending

A new layer belongs in DeepLearnLib when it speaks `dl::Tensor`, does not name a dataset, and does not encode a full network. A topology belongs in an application. See [Usage](../usage/models.md).

## Files

Declare the class in `include/DeepLearnLib/MyLayer.hpp`. Implement it in `src/MyLayer.cpp`. Add the `.cpp` to `DEEPLEARN_SOURCES` in `src/CMakeLists.txt`. If the file contains a `__global__` kernel, also add it to `DEEPLEARN_CUDA_SOURCES` so CMake compiles it as CUDA.

Implement `forward` and `backward`. With parameters, also implement `step`, `clip_gradients`, `get_parameters`, and `set_parameters`. Parameter names in the map are stable strings: `"weight"`, `"bias"`. `Network::save` and `load` round-trip that map.

## Contracts

Stay on the GPU. `forward` and `backward` do not call `to_host`.

Allocate with `ensure`, then reuse the slot:

```cpp
auto MyLayer::forward(const dl::Tensor& input, cudaStream_t stream) -> dl::Tensor
{
    const dl::StreamGuard guard(stream);
    dl::Tensor& output = dl::Tensor::ensure(
        output_cache_, input.get_shape(), dl::Device::GPU, input.get_dtype());
    // launch into output.data()
    return output.as_view();
}
```

`as_view()` shares the cache. A deep copy would allocate on every call.

Update weights in place:

```cpp
weights_.sgd_update_(weights_gradient_, step_learning_rate(), weight_decay, parameter_clip_bound());
```

When `Layer::momentum != 0`, call `sgd_momentum_update_` and keep the velocity in a member `optional<Tensor>`.

Convolution-style work uses the cuDNN C API. Dense products use `matmul_into`. Wrap calls in `CHECK_CUDNN`, `CHECK_CUBLAS`, or `CHECK_CUDA`. Throw `std::runtime_error` on a shape mismatch.

If `backward` depends on the forward input, store it during `forward`. `backward` cannot recompute from a tensor the caller has overwritten.

An elementwise op that always follows another op belongs in one kernel, or in a composite of the same kind as `FusedCBR2d`, rather than in a second global-memory pass.

## Tests

Add `tests/test_mylayer.cpp` to `dllib_tests`. Cover the output shape, a numeric check of `backward`, and `train()` versus `eval()` when they differ.

## Naming

Follow [Conventions](conventions.md): `PascalCase` type, `snake_case` methods, header file named after the type, private members with a trailing underscore. Put CUDA infrastructure in namespace `dl`. Leave the `Layer` subclass in the global namespace, next to `Conv2d` and `FullyConnected`.
