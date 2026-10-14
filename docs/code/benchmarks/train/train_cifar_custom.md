# train_cifar_custom.cpp

Trains `SimpleCNN` on CIFAR-10. The epoch loop, the CSV, and the schedule are `train_custom_classifier`, the same driver `train_cifar_torch` uses.

## main

1. Reads `cifar10_classification`.
2. Builds shuffled train and unshuffled test loaders on one class vocabulary.
3. Warns on stderr when the split is not 50,000 / 10,000 images.
4. Trains `SimpleCNN` and writes `metrics_custom.csv`, the confusion matrix, sample images, and `simplecnn_cifar10_final.bin`.
