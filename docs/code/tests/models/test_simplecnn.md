# test_simplecnn.cpp

SimpleCNN: logits sized to the class count, stack depth, and a single-channel input.

## ForwardLogitsMatchClassCount

A CIFAR batch of 2×3×32×32 and 10 classes yields finite logits and probabilities [2, 10] on the GPU.

## TrainableStackHasExpectedDepth

Conv–LeakyReLU–Pool twice, plus Flatten and FullyConnected, makes 8 layers; omitting the channel argument leaves in_channels at 3, and the class count and side come from the constructor (4 and 32).

## AcceptsSingleChannelMnistInput

Input [2, 1, 28, 28] with in_channels = 1 yields finite logits [2, 10].
