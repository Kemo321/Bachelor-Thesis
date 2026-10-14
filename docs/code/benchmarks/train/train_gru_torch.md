# train_gru_torch.cpp

Trains `TorchGruClassifier` on ECG200 with Adam. The splits, the shuffle seed, the gradient clip, and the CSV columns match `train_gru_custom`.

## main

1. Reads the same pipeline argument.
2. Writes `metrics_torch.csv` and `gru_classifier_torch_final.pt`.
