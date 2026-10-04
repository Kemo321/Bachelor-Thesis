# test_yolo.cpp

YOLOv1 model composition: backbone depth and detection-head width.

## ConstructsNonEmptyBackboneAndHead

The constructor builds 24 FusedCBR2d blocks and 4 pools, 28 backbone layers in total, and get_all_layers appends the head at the end.

## CustomClassCountChangesOutputWidth

Input 1×3×448×448 and two classes yield a finite tensor [1, 7·7·(10+2)] on the GPU.

## EvalModeForwardIsFinite

In eval, a 20-class model on a 448×448 input returns a finite tensor [1, 7·7·30].
