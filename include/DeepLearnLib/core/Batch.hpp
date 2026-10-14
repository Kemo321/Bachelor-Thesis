#pragma once

#include "DeepLearnLib/core/Tensor.hpp"

/**
 * One GPU-resident training or evaluation batch.
 *
 * ClassificationLoader uses images [N, C, H, W] and one-hot targets [N, Classes].
 * PackedImageLoader uses the same layout for a packed DLIMG001 file.
 */
struct Batch
{
    dl::Tensor images;
    dl::Tensor targets;
};
