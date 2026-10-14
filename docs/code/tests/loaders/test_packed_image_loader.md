# test_packed_image_loader.cpp

PackedImageLoader: the DLIMG001 format, float NCHW in [0, 1], and one-hot labels.

## YieldsNchwFloatAndOneHotLabels

Four 1×2×2 samples with two classes yield batch [2, 1, 2, 2]; pixels 0 and 255 become 0 and 1, and labels 0 and 1 become one-hot.

## RejectsBadMagic

A header other than "DLIMG001" is rejected by the constructor.
