# train_tabular_torch.cpp

Trains `TorchTabularMLP` on the same CSV, the same hold-out, and the same shuffle seed as `train_tabular_custom`.

## main

1. Loads the pipeline named by the first argument.
2. Trains with SGD.
3. Writes `metrics_torch.csv` and `tabular_mlp_torch_final.pt`.
