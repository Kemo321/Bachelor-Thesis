# train_mnist_torch.cpp

Trains an MNIST digit classifier in LibTorch, on the same data family as the custom-stack program. Epoch count, batch size, and learning rate come from the `mnist_classification` pipeline JSON. When the `epochs` key is missing, the code falls back to 10.

## main

1. Reads the configuration and the paths to `train.bin` and `test.bin`. Selects CUDA, or CPU when no device is present.
2. Opens a separate training loader (with shuffling) and a test loader. Image shape and class count come from the training file.
3. Builds `SimpleCNN`, moves it to the selected device, and creates SGD and LibTorch cross-entropy.
4. Writes the class names and opens the metrics CSV.
5. Each epoch sets the learning rate from the schedule and, on the training set, runs forward, backward, and an optimizer step. The batch is copied from a custom-stack tensor into a LibTorch tensor.
6. In eval mode it computes loss and accuracy on the test set. On the last epoch it collects the confusion matrix and up to 24 samples.
7. After the epoch it appends a CSV row. At the end it saves the confusion matrix, samples, and weights in the LibTorch format.

## SimpleCNNImpl

Registers two 3×3 convolutions with padding 1 (input channels → 16 → 32) and a linear layer on the map after two pooling stages. `forward` stacks LeakyReLU with slope 0.1, 2×2 max-pool, a flatten, and the linear layer into class logits.

## batch_to_torch

Copies images and one-hot labels from a custom-stack batch into LibTorch tensors on the given device. The class index is the argmax of the one-hot row.
