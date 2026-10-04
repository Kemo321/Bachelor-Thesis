# test_batchnorm2d.cpp

BatchNorm2d in train and eval: normalization, running statistics, the affine transform, and validation.

## TrainingNormalizesAndUpdatesRunningStats

Train on the batch {1, 3, 5, 7} yields an output with mean zero, and momentum 0.1 moves running_mean away from zero; the gradient has the input shape and is finite.

## EvalUsesRunningStatistics

In eval, values are multiplied by 1/√(running_var + ε) with gamma 1, beta 0, and variance 1.

## InvalidConstructorArgumentsThrow

The constructor rejects a non-positive channel count and a negative epsilon.

## ForwardRejectsChannelMismatchAndCpuInput

Forward requires the GPU, rank 4, and a channel count that matches the layer.

## BackwardRequiresTrainingForward

Backward in eval, and backward without a forward in train, throw.

## EvalAffineScaleAndShift

In eval, y = 2·x/√(1+ε) + 1 with gamma 2 and beta 1.

## ParameterRoundTripAndGpuTo

The readout returns gamma, beta, and the running statistics, and to(CPU) is rejected.

## StepChangesAffineParameters

After backward and step with learning rate 1, gamma differs from its value before the step.
