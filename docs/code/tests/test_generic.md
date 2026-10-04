# test_generic.cpp

MSE, cross-entropy, softmax, and CSVLoader.

## MseLossIsZeroWhenPredictionMatchesTarget

An identical prediction and target yield an MSE scalar of zero.

## MseLossDerivativeScalesByTwoOverN

The MSE gradient is 2·(prediction − target) / N; with a difference of 1 and N = 2 the result is 1.

## CrossEntropyPrefersTheCorrectOneHotClass

Logits whose maximum sits on the target class have a smaller loss than zero logits, and that loss stays positive.

## CrossEntropyGradientIsSoftmaxMinusTargetOverBatch

With zero logits and batch 1, softmax is [0.5, 0.5], so the gradient is softmax minus the one-hot target.

## SoftmaxRowsSumToOneAndBackwardIsFinite

Each softmax row sums to 1, and the input gradient is finite.

## SoftmaxRejectsCpuInputAndBackwardWithoutForward

Forward on the CPU, and backward without forward, throw.

## CsvLoaderReadsFeaturesAndTargets

The loader reads two feature columns and two target columns from a file with a header and lays them out as tensors [2, 2].

## MissingFileThrows

The CSVLoader constructor throws when the file does not exist.
