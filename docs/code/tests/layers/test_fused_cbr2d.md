# test_fused_cbr2d.cpp

The Conv–BatchNorm–LeakyReLU block: agreement with the split path in eval, finite training, the affine transform, and the parameter contract.

## EvalMatchesUnfusedConvBnLeaky

In eval, with identical weights and identity BatchNorm statistics, the fused block matches conv, then BN, then LeakyReLU.

## TrainingForwardBackwardAndStepAreFinite

Forward, backward, and step in train yield finite tensors, and the convolution weights change.

## AffineEvalScaleAndLeakySlope

The output is leaky(2·x/√(1+ε) − 1): the negative side is multiplied by the slope 0.1.

## InvalidConstructorArgumentsThrow

The constructor rejects non-positive channels or kernel, a negative epsilon, and BatchNorm momentum outside [0, 1].

## BackwardWithoutTrainingForwardThrows

Backward in eval, and backward without a forward in train, throw.

## ParameterRoundTrip

The parameter readout returns the weights, bias, gamma, beta, and statistics that were set; the default slope is 0.1, and to(CPU) is rejected.
