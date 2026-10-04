# train_mnist_custom.cpp

Trains an MNIST digit classifier on the custom stack (`SimpleCNN`, `dl::Tensor`). Epoch count, batch size, and learning rate come from the `mnist_classification` pipeline JSON. When the `epochs` key is missing, the code falls back to 10.

## main

1. Reads the configuration and the paths to the packed `train.bin` and `test.bin` files.
2. Opens a separate training loader (with shuffling) and a test loader. Class count, channels, and image side come from the file, and the two files must agree on those.
3. Builds `SimpleCNN`, moves the layers to the GPU, and attaches an SGD trainer with momentum, weight decay, and gradient clipping from the configuration.
4. Creates the results directory, writes the class names, and opens the metrics CSV.
5. Each epoch sets the learning rate from the schedule, computes cross-entropy, runs backpropagation, and takes an SGD step on the training set.
6. Switches the model to eval mode and computes loss and accuracy on the test set. On the last epoch it collects the confusion matrix and up to 24 sample predictions.
7. After the epoch it appends a CSV row. After the full run it saves the confusion matrix, image samples, and model weights.
