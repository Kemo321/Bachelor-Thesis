# test_network.cpp

A network of layers: forward, saving weights, fit, and construction checks.

## NullLayerThrows

A layer list that contains a null pointer is rejected by the constructor.

## ForwardThroughFlattenAndDense

Flatten of 1×1×2×2 and a dense layer 4→3 whose weights select the first three features yield [1, 2, 3].

## SaveLoadRoundTripRestoresWeights

Saving and loading a weight file restores the same weights and bias.

## LoadMissingFileThrows

load on a path that does not exist throws.

## LoadLayerCountMismatchThrows

A one-layer checkpoint fails to load into a two-layer network.

## FitNegativeEpochsThrows

A negative epoch count is rejected.

## FitOneEpochOnYoloShapedOutput

One epoch on a head of width 7·7·30 (the YOLOv1 grid for 20 classes) leaves a finite prediction [1, 7·7·30].

## AssignsLearningRateToLayers

The learning rate passed to the network overwrites the layer learning rate.
