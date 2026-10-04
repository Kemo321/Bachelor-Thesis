# test_numerics.cpp

Finite losses for bad boxes and large logits, and gradient clipping.

## YoloLossStaysFiniteForZeroAndNegativeBoxSizes

A width of 0 and a negative height on the first box leave the loss and the gradient finite.

## CrossEntropyStaysFiniteForOverflowingLogits

Logits on the order of 1000, where exp overflows float unless the maximum is subtracted, yield a finite non-negative loss and a finite gradient.

## CrossEntropyClampsNearZeroProbabilities

When the target-class probability falls to zero, the log is clamped and the loss stays near −log(kSafeEps).

## NetworkStoresConfigurableGradientClip

The network stores the given clip threshold, and the setter replaces it.

## ClipLossGradientBoundsEveryElement

With the default loss scale of 1, threshold 3 clips 100 and −50, and leaves 0 and 3 unchanged.

## ParameterGradientClipBoundsDenseUpdates

A weight gradient of 1000, with clip threshold 1 and learning rate 1, updates the weight to about −1.

## HasNonFiniteDetectsInfAndNan

has_non_finite detects Inf and NaN, and a finite tensor returns false.

## DefaultNetworkGradientClipIsDisabled

The constructor without a threshold sets clipping to 0, which turns clipping off.
