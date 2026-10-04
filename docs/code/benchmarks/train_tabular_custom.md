# train_tabular_custom.cpp

Trains a small dense network on a CSV table, on the custom stack (`FullyConnected`, `LeakyReLU`). The pipeline name is the program's first argument, and without an argument it is `tabular_demo`. Epoch count, hidden-layer size, and the CSV path come from that pipeline's JSON. When the `epochs` key is missing, the code falls back to 20.

## main

1. Reads the configuration, class names, and paths. Missing class names are replaced with numbers.
2. When the CSV file is missing, it writes a dummy table. It then loads every row into host memory.
3. Assembles two dense layers with LeakyReLU and a separate softmax for reading probabilities. The trainable layers go to the GPU, and softmax stays in eval mode.
4. Opens the metrics CSV and prepares a random row order and a confusion matrix.
5. Each epoch shuffles the rows, sets the learning rate from the schedule, and on every batch computes cross-entropy, accuracy, and the confusion matrix, then takes an SGD step.
6. After the epoch it appends a CSV row with loss, time, and accuracy. After the last epoch it saves the confusion matrix. What lands on disk is those two files plus the class names.

## pipeline_name_from_args

Returns the pipeline name from the program's first argument. An empty or missing argument yields `tabular_demo`, so the default tabular experiment can be launched.
