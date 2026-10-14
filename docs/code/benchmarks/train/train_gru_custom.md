# train_gru_custom.cpp

Trains `GruClassifier` on ECG200 with Adam. The epoch driver is `train_custom_rows`, the same one the tabular MLP uses. The Torch twin is `train_gru_torch`.

## main

1. Reads the pipeline named by the first argument (`ecg200` when it is omitted).
2. Loads the TRAIN and TEST splits and stops when their lengths or class names differ.
3. Shuffles the training order with seed 42 once per epoch. The test order stays identity.
4. Writes `metrics_custom.csv` and `gru_classifier_final.bin`.
