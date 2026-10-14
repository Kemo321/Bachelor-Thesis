# train_cifar_torch.cpp

Trains `TorchSimpleCNN` on CIFAR-10. The loader, the epoch count, the schedule, and the CSV columns match `train_cifar_custom`.

## main

1. Reads `cifar10_classification`.
2. Builds the same train and test loaders as the custom binary.
3. Trains with SGD and writes `metrics_torch.csv`, the confusion matrix, sample images, and `simplecnn_cifar10_torch_final.pt`.
