# test_activations.cpp

Forward and backward for LeakyReLU and Dropout on the GPU, including evaluation mode and rejection of bad arguments.

## LeakyReLUForwardAndBackwardSlope

Negative inputs are multiplied by the slope 0.1, and the local gradient is 0.1 for x ≤ 0 and 1 for x > 0.

## DropoutEvalPreservesValues

In eval, Dropout is the identity for values and for the gradient.

## DropoutTrainingZerosAndScales

In training with p = 0.5 some values vanish, and the kept values are scaled by 1/(1−p) = 2.

## LeakyReLUDefaultSlopeOnNegatives

The default slope 0.1 scales a negative vector.

## LeakyReLUZeroSlopeIsRelu

A slope of 0 zeros negative values and their gradient, which matches ReLU.

## LeakyReLURejectsCpuAndMismatchedBackward

Forward rejects the CPU, and backward rejects a wrong gradient size and a missing earlier forward.

## DropoutZeroProbabilityKeepsValues

A drop probability of 0 in train leaves the values unscaled.

## DropoutInvalidProbabilityThrows

The constructor rejects a probability outside [0, 1).

## DropoutRejectsCpuAndMismatchedBackward

Forward rejects the CPU, and backward rejects a gradient of a different shape.

## DropoutTrainThenEvalDisablesMask

After the switch to eval, the training mask is not applied and ones pass through unchanged.
