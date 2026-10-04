# test_profiler_map.cpp

mAP and IoU on the CPU, and Profiler on the GPU.

## PerfectOverlapYieldsUnitMap

An identical prediction box and ground-truth box at IoU threshold 0.5 yield mAP equal to 1.

## DisjointBoxesYieldZeroMap

Disjoint boxes produce no true positive, so mAP is 0.

## EmptyGroundTruthIsZeroAndInvalidThresholdThrows

Missing ground truth yields mAP 0, and threshold 1.5 lies outside [0, 1] and throws.

## DetectionIouIsOneForIdenticalBoxes

The IoU of two copies of the same box is 1.

## StopReturnsNonNegativeGpuMilliseconds

The time between start and stop is finite and non-negative, and the sum of 1024 ones is 1024.

## VramUsageIsReportedInMebibytes

After a GPU allocation, the reported VRAM use in MiB is greater than zero.

## StopWithoutStartThrows

stop() without start() throws.
