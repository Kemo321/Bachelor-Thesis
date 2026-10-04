# test_fullyconnected.cpp

The FullyConnected layer: a matrix product with bias, an SGD step, and shape validation.

## ForwardMatchesMatmulPlusBias

The output is input @ weights + bias and has shape [1, 3].

## BackwardAndStepUpdateWeights

With zero weights, learning rate 1, and a gradient equal to the identity, the weights after step are −I.

## InvalidSizesThrow

The constructor rejects a non-positive input or output feature count.

## ForwardRejectsWrongRankDeviceAndFeatures

Forward requires the GPU, rank 2, and a feature count that matches the layer.

## BackwardWithoutForwardThrows

Backward without forward throws.

## SecondBackwardWithoutForwardThrows

A second backward without a new forward is rejected.

## BackwardRejectsBatchMismatch

A gradient whose batch differs from the remembered forward is rejected.

## ZeroWeightsReturnBias

With zero weights, every output row equals the bias.

## ParameterRoundTripAndGpuTo

The readout returns the weights and bias that were set, and to(CPU) is rejected.

## DefaultInitializationIsFinite

A fresh 8→4 layer has finite weights [8, 4] and bias [1, 4].
