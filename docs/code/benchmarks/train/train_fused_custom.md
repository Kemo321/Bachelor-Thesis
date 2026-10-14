# train_fused_custom.cpp

Trains `FusedCbrCNN` on CIFAR-10 through `train_custom_classifier`. The pipeline key is `cifar10_fused`. The CSV schema matches the other training programs.

## main

1. Reads `cifar10_fused` and the CIFAR-10 folder layout.
2. Warns on stderr when the split is not 50,000 / 10,000 images.
3. Writes `results/cifar10_fused/metrics_custom.csv` and `fused_cbr_cnn_cifar10_final.bin`.
