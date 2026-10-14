# test_maxpool2d.cpp

MaxPool2d: argmax, several windows, channels, and shape validation.

## DownsamplesAndRoutesGradientToArgmax

A 2×2 pool of the window {1, 3, 2, 0} returns 3, and a gradient of 1 comes back only at the maximum.

## FourByFourWindows

Four 2×2 windows on a 4×4 map leave the maxima 6, 8, 9, and 7, and the gradient sum is 4.

## InvalidConstructorArgumentsThrow

A non-positive window size or stride is rejected by the constructor.

## ForwardRejectsCpuAndNonNchw

Forward requires the GPU and rank 4 (NCHW).

## BackwardWithoutForwardThrows

Backward without forward throws.

## BackwardRejectsMismatchedGradientShape

A gradient with the input shape is rejected; backward expects the pooled output shape.

## MultiChannelPoolsIndependently

Each channel keeps its own maximum: 1 and 5.

## RepeatedForwardReusesDescriptors

Two successive forwards on the same shape return the maximum of the current window (here 4 and 4).
