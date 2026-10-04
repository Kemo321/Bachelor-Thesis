# SimpleCNN.cpp

A small classification network built from DeepLearnLib layers. `forward` and `get_all_layers` keep those names. Softmax is a separate object: cross-entropy training takes logits from `forward_logits`, and `get_all_layers` does not include it.

## conv_out

Spatial size after one layer, integer division: `(input + 2 * padding - kernel) / stride + 1`. `flatten_features` uses this formula.

## flatten_features

Computes the input width of the last FC layer for a given image side. Two `3×3` convolutions with padding 1 keep the side. Each `2×2` pooling with stride 2 uses `conv_out`. The last map has 32 channels, so the result is `32 * side * side`. For a 32×32 image the side goes 32, 16, 8, and the FC width is `32*8*8`.

## SimpleCNN::SimpleCNN

Checks that the class count, side, and input channel count are positive. Then it fills `layers_`:

1. `Conv2d` from `in_channels` to 16, kernel `3×3`, stride 1, padding 1.
2. `LeakyReLU(0.1)`.
3. `MaxPool2d` `2×2` with stride 2.
4. `Conv2d` 16→32, kernel `3×3`, padding 1.
5. `LeakyReLU(0.1)`.
6. `MaxPool2d` `2×2`.
7. `Flatten`.
8. `FullyConnected` from `flatten_features(image_size)` to `num_classes`.

`Softmax` is created beside this list and is not appended to it.

## SimpleCNN::forward_logits

Binds cuDNN to the stream and walks `layers_` in constructor order. After every layer, `view` keeps the shape. The result is logits, without Softmax.

## SimpleCNN::forward

Calls `forward_logits`, then `Softmax::forward` on the same stream. Returns class probabilities.

## SimpleCNN::get_all_layers

Returns `layers_` in `forward_logits` order. Softmax stays off the list, so the optimizer step and the weight save do not include it.

## SimpleCNN::num_classes

Returns the class count given to the constructor. The last linear layer and the softmax axis have this width.

## SimpleCNN::image_size

Returns the image side given to the constructor. `flatten_features` depends on it when the network is built.

## SimpleCNN::in_channels

Returns the input channel count given to the constructor. The first convolution has this depth.
