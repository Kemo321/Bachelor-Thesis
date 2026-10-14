# test_flatten.cpp

The Flatten layer: keeping the batch, collapsing the remaining axes, and rejecting bad inputs.

## ReshapesAndUnflattensGradient

A 2×3×2×2 input collapses to [2, 12], and backward restores the NCHW shape and the same values.

## AlreadyRankTwoIsUnchanged

A rank-2 tensor [batch, features] passes through with its shape unchanged.

## FiveDimensionalInput

A 2×2×2×2×2 input collapses to [2, 16].

## ScalarAndCpuInputsThrow

A scalar and a CPU tensor are rejected.

## BackwardWithoutForwardThrows

Backward without an earlier forward throws.

## OneDimensionalTreatsLengthAsBatch

A vector of length 4 is treated as batch 4 and one feature, that is shape [4, 1].
