#pragma once

#include "DeepLearnLib/Layer.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct DarknetLoadOptions
{
    /** Load at most this many convolutional / FusedCBR blocks. `0` means all. */
    int cutoff_convs { 0 };
    /** Load `[local]` if present in the file and the model. */
    bool load_local { true };
    /** Load `[connected]` layers. */
    bool load_connected { true };
};

struct DarknetLoadReport
{
    int major { 0 };
    int minor { 0 };
    int revision { 0 };
    std::uint64_t seen { 0 };
    bool transpose_connected { false };
    int convs_loaded { 0 };
    int locals_loaded { 0 };
    int connected_loaded { 0 };
    std::size_t bytes_remaining { 0 };
};

/**
 * Load a Darknet `.weights` stream into Conv2d / FusedCBR2d / LocalLayer / FullyConnected.
 *
 * Convolutional-with-BN (yolov1.cfg): `bias, scale, rolling_mean, rolling_var, W[n,c,k,k]`.
 * Local: `bias[out_h*out_w*n], W[locations, n, c, k, k]`.
 * Connected: `bias[out], W[out, in]` transposed into FullyConnected `[in, out]` unless the
 * file header requests the already-transposed layout (`major>1000 || minor>1000`).
 */
auto load_darknet_weights(const std::vector<std::shared_ptr<Layer>>& layers, const std::string& path,
    const DarknetLoadOptions& options = {}) -> DarknetLoadReport;
