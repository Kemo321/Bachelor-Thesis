# test_conv2d.cpp

Conv2d on the GPU: output shape, window sum, backward, and validation.

## ForwardOutputShape

Padding 1 with a 3×3 kernel and stride 1 keeps the side at 16, and 8 filters yield NCHW 2×8×16×16.

## OnesKernelMatchesWindowSum

A 2×2 kernel of ones on an input of ones sums each window to 4 and yields a 2×2 map.

## IdentityOneByOneAndBackward

A 1×1 convolution of weight 1 copies the input, and a unit gradient comes back as ones of the same shape.

## BackwardWithoutForwardThrows

Backward without an earlier forward throws.

## InvalidConstructorArgumentsThrow

The constructor rejects non-positive channels, kernel, or stride, and negative padding.

## ForwardRejectsChannelMismatchAndCpuInput

Forward requires the GPU, rank 4, and a channel count that matches the layer.

## StrideTwoHalvesSpatialSize

A 1×1 kernel with stride 2 on a 4×4 map yields a spatial size of 2×2.

## ZeroWeightsAddBias

With zero weights, every output equals the bias.

## ParameterRoundTripAndGpuTo

The readout returns the weights and bias that were set, and to(CPU) is rejected.

## StepUpdatesWeightsAfterBackward

After backward and step with learning rate 1, the 1×1 weight changes.

## BackwardRejectsMismatchedGradientShape

A gradient whose spatial shape differs from the forward output is rejected.
