# YOLOv1Impl.cpp

Factory for a shortened network used with `Network::forward`. The detector used in the benchmarks and inference is the `YOLO` class in `models/YOLO.cpp`: that file has the full backbone and a `Flatten` layer.

## create_yolo_v1

`create_yolo_v1` returns a `Network` with learning rate `0.001`. The network constructor copies that rate into the layers.

`forward` of this network walks the list in order:

1. Input block: `Conv2d` `3→64`, kernel `7×7`, stride 2, padding 3, then `LeakyReLU(0.1)` and `MaxPool2d` `2×2`.
2. Second block: `Conv2d` `64→192`, kernel `3×3`, stride 1, padding 1, then `LeakyReLU(0.1)` and `MaxPool2d` `2×2`.
3. Linear head: `FullyConnected` `7*7*1024 → 4096`, `LeakyReLU(0.1)`, `FullyConnected` `4096 → 7*7*30`.

For a 448 input the first convolution yields side 224, pooling yields 112, the second convolution keeps 112 at 192 channels, and the second pooling goes down to 56. The first FC layer still declares an input of `7*7*1024`. This list has neither the convolutions that would produce that tensor nor a `Flatten` layer. `FullyConnected` requires a rank-2 tensor.
