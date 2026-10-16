#pragma once

#include "DeepLearnLib/core/Tensor.hpp"

struct Batch
{
    dl::Tensor images;
    dl::Tensor targets;
};
