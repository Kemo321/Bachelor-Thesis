# test_loss.cpp

YOLOLoss on a 7×7 grid: an empty grid, an object, the gradient, and rejection of a bad layout. A cell has 30 values: two boxes of 5 plus 20 classes.

## LossOnEmptyGridIsNearZeroScalar

An empty prediction and an empty target yield a finite GPU scalar near zero, within 1e-4.

## LossIsPositiveWhenPredictionMissesAnObject

A target with an object in cell (3, 3) and a zero prediction yield a finite positive loss.

## MatchingObjectPredictionHasSmallerLossThanZeros

A prediction that matches the target has a smaller finite loss than a zero prediction.

## LossDerivativeMatchesPredictionShapeAndIsFinite

The gradient has the prediction's shape, is finite, and is not entirely zero.

## FlattenedLayoutIsAcceptedAndGradientsMatchRank

Layout [N, 7·7·30] is accepted: the loss is a scalar, and the gradient stays rank 2.

## BatchSizeMismatchThrows

A mismatch between the prediction batch and the target batch is rejected in loss and in loss_derivative.

## InvalidClassCountThrows

A non-positive class count is rejected in loss and in loss_derivative.

## CpuTensorsThrow

CPU tensors are rejected.

## RankThreeAndWrongGridThrow

Rank 3, a grid other than 7×7, and a flattened tensor of the wrong width are rejected.

## SingleClassGridIsAccepted

An empty grid with one class (11 attributes per cell) yields a loss near zero within 1e-4 and a gradient with the prediction's shape.

## TwoObjectsYieldFinitePositiveLoss

Two occupied cells and a zero prediction yield a finite positive loss.

## MixedRankLayoutsAreAcceptedTogether

A rank-4 target and a flattened prediction of the same batch yield a finite scalar.
