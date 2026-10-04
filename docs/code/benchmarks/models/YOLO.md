# YOLO.cpp

YOLOv1 detector built from DeepLearnLib layers. The input for which the head is meaningful is NCHW `3×448×448`: the last FC layers hard-code a `7×7` grid. The class count enters only the last layer, as `7*7*(10 + num_classes)` (two boxes of 5 values each, plus the classes).

`forward` and `get_all_layers` keep those names. Training walks the list from `get_all_layers` backward.

## YOLO::YOLO

The constructor builds the backbone and the head. Each backbone block is a `FusedCBR2d`: convolution, batch-norm, and LeakyReLU with slope `0.1`.

For a 448 input the stages produce these sides:

1. Convolution `7×7`, stride 2, padding 3 (`3→64`) yields 224, and `2×2` pooling goes down to 112.
2. Convolution `3×3` (`64→192`) stays on 112, and pooling goes down to 56.
3. Four convolutions (`192→128→256→256→512`, alternating `1×1` and `3×3`) stay on 56, and pooling goes down to 28.
4. Four times the pair `512→256` (`1×1`) and `256→512` (`3×3`), then `512→512` (`1×1`) and `512→1024` (`3×3`). Pooling goes down to 14.
5. Twice the pair `1024→512` and `512→1024`, then a `3×3` convolution (`1024→1024`), a `3×3` convolution with stride 2 (side 14 goes down to 7), and two `3×3` convolutions that stay on `7×7` at 1024 channels.

The head, already on `7×7×1024`: `Flatten`, `FullyConnected` `7*7*1024 → 4096` with `inertia` equal to 0 (in this layer that is the GEMM beta when writing the weight gradient; SGD momentum is in `Layer::momentum`), `LeakyReLU(0.1)`, `Dropout(0.5)`, `FullyConnected` `4096 → 7*7*(10 + num_classes)`.

## YOLO::forward

`forward` binds cuDNN to the given stream and walks `backbone_layers` first, then `head_layers`. After every layer, `view` keeps the current shape. The function does not switch train/eval mode: dropout in the head depends on the state set on the layers earlier.

## YOLO::get_all_layers

Returns a new vector: the whole backbone, then the whole head, in `forward` order. The training loop calls `backward` from the end of this list, and the network weight save takes it as the parameter list.
