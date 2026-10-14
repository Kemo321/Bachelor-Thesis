# test_tensor.cpp

Allocation, strides, host–device copies, GEMM, and elementwise tensor operations.

## CpuAllocationAndZeroInitialization

A 2×3×4 shape on the CPU has 24 zeroed elements.

## ScalarCpuAllocation

An empty shape is a scalar: one element and a non-null pointer.

## GpuAllocationBehavior

Constructing 10×10 on the GPU succeeds when a CUDA device is present, and throws otherwise.

## ViewConstructorSharesMemoryOwnership

A view over a shared_ptr points at the same memory and raises the reference count to 2; after the view is destroyed it is 1.

## OneDimensionalAllocationShapeAndStrides, TwoDimensionalAllocationShapeAndStrides, ThreeDimensionalAllocationShapeAndStrides

GPU allocation sets the size to the product of the dimensions and row-major strides: {1}, {4, 1}, and {12, 4, 1}.

## NchwStridesMatchNetworkLayout

For 1×3×8×8 the NCHW strides are {192, 64, 8, 1}, that is C·H·W, H·W, W, 1.

## FromHostVectorRoundTrip

A host vector uploaded to the GPU comes back with the same shape and values.

## FromHostPointerRoundTrip

A pointer to four floats comes back from the GPU with the same values.

## FromHostCpuDeviceRoundTrip

from_host with Device::CPU leaves the tensor on the CPU.

## FromHostRejectsMismatchedBufferSize

A buffer shorter than the product of the shape is rejected.

## FromHostRejectsNullPointer

A null host pointer is rejected.

## ZerosLikeMatchesShapeDeviceAndValues

zeros_like copies the shape and device and fills zeros.

## SquareMatmul

The product [[1, 2], [3, 4]] · [[5, 6], [7, 8]] is [[19, 22], [43, 50]].

## RectangularMatmul

A 2×3 matrix times 3×4 yields 2×4 with values 38, 44, 50, 56 and 83, 98, 113, 128.

## LogicalTransposeMatmulMatchesPhysicalTranspose

The left-argument transpose flag matches an explicit transpose() before the multiply, with shape [3, 4].

## LogicalTransposeBMatmulMatchesPhysicalTranspose

The right-argument transpose flag matches an explicit transpose(), with shape [2, 4].

## InPlaceAddMulAndAddScaledDoNotAllocateTemps

add_, mul_, and add_scaled_ keep the same pointer; after (a+b)·0.5 + 0.1·b the values are 6.5, 13, 19.5, 26.

## MatmulIntoAccumulatesIntoExistingBuffer

matmul_into with beta 1 adds the product onto an existing buffer of ones, so 19 becomes 20.

## AddRowBroadcastsBiasAcrossBatch

Bias [0.5, −1, 2] is added to every row of a 2×3 matrix.

## AddSumRowsAccumulatesWithBeta

acc[j] = 0.5·10 + the sum of column j, that is 10, 12, and 14.

## IdentityMatmulLeavesMatrixUnchanged

Multiplying by the identity matrix returns the same values.

## MatmulMismatchedInnerDimensionsThrows

A mismatched inner multiply dimension is rejected.

## ElementwiseTensorAddition

Elementwise addition of two 2×2 matrices sums corresponding values.

## ElementwiseTensorSubtraction

Elementwise subtraction computes the difference of corresponding values.

## ElementwiseTensorMultiplication

Elementwise multiplication computes the product of corresponding values.

## TensorScalarMultiplicationAndAddition

Scalar multiplication and addition act on every element.

## ElementwiseSizeMismatchThrows

A different operand size is rejected for +, −, and *.

## ElementwiseCpuOperandThrows

Adding a GPU tensor with a CPU tensor, and scalar multiplication on the CPU, are rejected.

## ClampBoundsValues

clamp clips values to the given interval.

## ClampInvalidRangeThrows

A lower bound greater than the upper bound is rejected.

## GlobalSumPositiveValues

The sum of all positive elements is a scalar equal to their sum.

## GlobalSumNegativeAndMixedValues

sum(−1) on {1, −2, 3, −4.5} yields the scalar −2.5.

## SumAlongAxisThrows

sum(0) on a vector is rejected.

## ViewReshapeSharesStorageAndValues

view onto a compatible shape shares the pointer and the values with the original.

## ViewInfersMinusOneDimension

A single axis of −1 is inferred from the product of the others.

## ViewIncompatibleShapeThrows

A shape with a different element count, and two axes of −1, are rejected.

## TransposeTwoDimensional

Transposing a matrix swaps the dimensions and lays the elements out by columns.

## TransposeThenMatmulMatchesOriginalProductLayout

Transposing the matrix [[1, 2], [3, 4]] lays the host out as 1, 3, 2, 4.

## TransposeRejectsNonTwoDimensionalTensors

transpose() rejects a vector and a tensor whose rank is not 2.

## VectorMatrixMatmul

The vector [1, 2, 3] times a 3×2 matrix yields [4, 5].

## MatmulOnCpuThrows

matmul on the CPU is rejected.

## ChainedElementwiseOps

(a + b) · c is computed elementwise.

## ClampEqualBoundsSaturates

clamp with equal bounds sets every element to that bound.

## ViewFlattensWithMinusOne

view({−1}) flattens the tensor to one dimension whose length equals the element count.

## ViewRejectsInvalidNegativeDimension

Axis −2 in view is rejected.

## ViewIdentityKeepsShape

view of the same shape returns the same pointer and shape.

## SumOfZerosIsZero

The sum of a zero tensor is zero.

## TransposeOneByN

A 1×N row becomes an N×1 column after transposition.

## MoveConstructorTransfersOwnership

After the move, the new tensor has the same pointer and the values 7 and 8.

## ZerosLikeCpuSourceStaysOnCpu

zeros_like of a CPU source stays on the CPU and is zero.

## CpuToHostRoundTrip

to_host on a CPU tensor returns the stored values.
