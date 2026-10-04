# train_cifar_torch.cpp

Trains a CIFAR-10 classifier in LibTorch. Epoch count, batch size, image side, and split names come from the `cifar10_classification` pipeline JSON. When the `epochs` key is missing, the code falls back to 20.

## main

1. Reads the configuration and the dataset and results directories. Selects CUDA, or CPU when no device is present.
2. Opens a training loader with shuffling and a test loader on a shared class vocabulary. It stops when the class counts differ.
3. Builds `SimpleCNN` with three input channels, moves it to the selected device, and creates SGD and cross-entropy.
4. Writes the class names and opens the metrics CSV.
5. Each epoch sets the learning rate from the schedule and, on the training set, runs forward, backward, and an optimizer step.
6. In eval mode it computes loss and accuracy on the test set. On the last epoch it collects the confusion matrix and up to 24 samples.
7. After the epoch it appends a CSV row. At the end it saves the confusion matrix, samples, and weights in the LibTorch format.

## SimpleCNNImpl

Registers two 3×3 convolutions with padding 1 (3 → 16 → 32 channels) and a linear layer on the map after two pooling stages. `forward` stacks LeakyReLU with slope 0.1, 2×2 max-pool, a flatten, and the linear layer into class logits.

## batch_to_torch

Copies images and one-hot labels from a custom-stack batch into LibTorch tensors on the given device. The class index is the argmax of the one-hot row.
