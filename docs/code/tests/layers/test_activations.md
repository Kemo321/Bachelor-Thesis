# test_activations.cpp

Forward and backward for LeakyReLU on the GPU, including rejection of bad arguments.

## LeakyReLUForwardAndBackwardSlope

Negative inputs are multiplied by the slope 0.1, and the local gradient is 0.1 for x ≤ 0 and 1 for x > 0.

## LeakyReLUDefaultSlopeOnNegatives

The default slope 0.1 scales a negative vector.

## LeakyReLUZeroSlopeIsRelu

A slope of 0 zeros negative values and their gradient, which matches ReLU.

## LeakyReLURejectsCpuAndMismatchedBackward

Forward rejects the CPU, and backward rejects a wrong gradient size and a missing earlier forward.
