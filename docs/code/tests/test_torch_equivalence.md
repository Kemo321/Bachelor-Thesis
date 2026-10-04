# test_torch_equivalence.cpp

Forward and backward agreement with LibTorch within kTorchTol (1e-3).

## Conv2dForwardAndBackwardMatchLibTorch

The same weights and upstream gradient yield activations and dX that match torch::conv2d within the LibTorch tolerance 1e-3.

## MaxPool2dForwardAndBackwardMatchLibTorch

A 2×2 pool on unique values routes the gradient the same way as torch::max_pool2d, within the LibTorch tolerance 1e-3.

## FullyConnectedForwardAndBackwardMatchLibTorch

Y = XW + b matches torch::linear after transposing LibTorch weights from layout [out, in], within the LibTorch tolerance 1e-3.

## BatchNorm2dTrainForwardAndBackwardMatchLibTorch

Train BatchNorm with eps 1e-5 and momentum 0.1 matches torch::batch_norm at those same constants, within the LibTorch tolerance 1e-3.

## FusedCBR2dEvalForwardMatchesLibTorch

Eval fused matches conv, batch_norm (eps 1e-5), and leaky_relu(0.1), within the LibTorch tolerance 1e-3.

## YOLOLossMatchesLibTorchBaseline

The YOLOv1 loss and gradient match the LibTorch baseline within 1e-3; the gradient comparison detaches box assignment, and box sizes are at least 0.05.
