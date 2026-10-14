#pragma once

/**
 * Spatial size after one convolution or pool: (input + 2 * padding - kernel) / stride + 1.
 * Integer division matches both cuDNN and LibTorch for these exact sizes.
 */
inline auto conv_spatial(int size, int kernel, int stride, int padding) -> int
{
    return ((size + (2 * padding) - kernel) / stride) + 1;
}

/**
 * Flattened width after two 3×3 convolutions with padding 1 and two 2×2 pools.
 * The last map has `channels` channels (32 in SimpleCNN and FusedCbrCNN).
 */
inline auto pooled_cnn_features(int image_size, int channels = 32) -> int
{
    int spatial = conv_spatial(image_size, 3, 1, 1);
    spatial = conv_spatial(spatial, 2, 2, 0);
    spatial = conv_spatial(spatial, 3, 1, 1);
    spatial = conv_spatial(spatial, 2, 2, 0);
    return channels * spatial * spatial;
}
