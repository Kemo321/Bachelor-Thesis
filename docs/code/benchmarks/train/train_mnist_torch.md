# train_mnist_torch.cpp

Trains `TorchSimpleCNN` on the packed MNIST files. The checks and the CSV columns match `train_mnist_custom`.

## main

1. Reads `mnist_classification`.
2. Builds `TorchSimpleCNN` from the packed channel count and side.
3. Writes `metrics_torch.csv` and `simplecnn_mnist_torch_final.pt`.
