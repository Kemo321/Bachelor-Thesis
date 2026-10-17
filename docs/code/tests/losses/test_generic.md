# test_generic.cpp

Cross-entropy and CSVLoader.

## CrossEntropyPrefersTheCorrectOneHotClass

Logits whose maximum sits on the target class have a smaller loss than zero logits, and that loss stays positive.

## CrossEntropyGradientIsSoftmaxMinusTargetOverBatch

With zero logits and batch 1, softmax is [0.5, 0.5], so the gradient is softmax minus the one-hot target.

## CsvLoaderReadsFeaturesAndTargets

The loader reads two feature columns and two target columns from a file with a header and lays them out as tensors [2, 2].

## MissingFileThrows

The CSVLoader constructor throws when the file does not exist.
