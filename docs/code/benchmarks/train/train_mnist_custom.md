# train_mnist_custom.cpp

Trains `SimpleCNN` on the packed MNIST files. The epoch driver is `train_custom_classifier`, shared with CIFAR-10.

## main

1. Reads `mnist_classification` and opens `train.bin` / `test.bin`.
2. Stops when the two files disagree on channels, spatial size, or class count.
3. Warns on stderr when the split is not 60,000 / 10,000 images.
4. Writes `metrics_custom.csv` and `simplecnn_mnist_final.bin`.
