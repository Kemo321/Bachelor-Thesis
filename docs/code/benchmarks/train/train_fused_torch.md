# train_fused_torch.cpp

Trains `TorchFusedCbrCNN` on CIFAR-10. The loader, the schedule, and the CSV columns match `train_fused_custom`.

## main

1. Reads `cifar10_fused`.
2. Builds Conv-BatchNorm-LeakyReLU blocks with epsilon `1e-5` and momentum `0.1`.
3. Writes `metrics_torch.csv` and `fused_cbr_cnn_cifar10_torch_final.pt`.
