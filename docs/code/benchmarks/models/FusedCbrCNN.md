# FusedCbrCNN.cpp

CIFAR-style CNN whose convolution blocks are `FusedCBR2d` (convolution, batch-norm, LeakyReLU 0.1). Each block is followed by `MaxPool2d(2, 2)`. `Flatten` and `FullyConnected` form the head. The flatten width is `pooled_cnn_features` from `cnn_shapes.hpp`, the same formula `SimpleCNN` uses.

Batch-norm epsilon is `1e-5` and the new-batch blend is `0.1`, matching `TorchFusedCbrCNN`.
