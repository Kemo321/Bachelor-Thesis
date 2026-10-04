# train_cifar_custom.cpp

Trains a CIFAR-10 classifier on the custom stack (`SimpleCNN`, `dl::Tensor`). Epoch count, batch size, image side, and split names come from the `cifar10_classification` pipeline JSON. When the `epochs` key is missing, the code falls back to 20.

## main

1. Reads the configuration and the dataset and results directories.
2. Opens a training loader with shuffling and a test loader on the same class vocabulary. It stops when the class counts differ, and warns when the set sizes depart from 50,000 and 10,000 images.
3. Builds `SimpleCNN`, moves the layers to the GPU, and attaches an SGD trainer.
4. Creates the results directory, writes the class names, and opens the metrics CSV.
5. Each epoch sets the learning rate from the schedule, computes cross-entropy, and updates the weights on the training set. Every 50 batches it writes a diagnostic log.
6. Switches the model to eval mode and computes loss and accuracy on the test set. On the last epoch it collects the confusion matrix and up to 24 samples.
7. After the epoch it appends a CSV row. After the full run it saves the confusion matrix, samples, and model weights.
