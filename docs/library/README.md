# DeepLearnLib

Public API of the library (`include/DeepLearnLib/`, `src/`). Both trees use `core/`, `layers/`, `losses/`, `loaders/`, and `utils/`. This manual stops at the types you call. Epoch loops, dataset schedules, and Custom-vs-Torch binaries are in [Usage](../usage/README.md).

| Page | Subject |
| --- | --- |
| [Overview](overview.md) | What the library owns, and the class diagram |
| [Tensor](tensor.md) | Storage, views, `ensure`, GEMM |
| [Layers](layers.md) | `Layer` and every subclass |
| [Losses](losses.md) | `CrossEntropyLoss`, `MSELoss` |
| [Loaders](loaders.md) | How a `Batch` is produced |
| [Utilities](utilities.md) | `Network`, precision, profiler, logging |
| [Extending](extending.md) | Adding a layer |
| [Build](build.md) | Configuring and compiling the library |
| [Conventions](conventions.md) | Names and the exceptions |

Dependency direction inside the library:

```mermaid
flowchart TB
    subgraph types [Types callers use]
        layer[Layer and subclasses]
        loss[Losses]
        loaders[Loaders]
        network[Network]
    end
    tensor["dl::Tensor"]
    vendor["CUDA, cuBLAS, cuDNN"]

    layer --> tensor
    loss --> tensor
    loaders --> tensor
    network --> layer
    network --> loss
    tensor --> vendor
    layer --> vendor
```

`SimpleCNN` is not in this diagram. It lives in `DeepLearnModels` and is documented under Usage. The tabular MLP and the GRU head are local to their training programs.
