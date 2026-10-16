#pragma once

inline auto conv_spatial(int size, int kernel, int stride, int padding) -> int
{
    return ((size + (2 * padding) - kernel) / stride) + 1;
}

inline auto pooled_cnn_features(int image_size, int channels = 32) -> int
{
    int spatial = conv_spatial(image_size, 3, 1, 1);
    spatial = conv_spatial(spatial, 2, 2, 0);
    spatial = conv_spatial(spatial, 3, 1, 1);
    spatial = conv_spatial(spatial, 2, 2, 0);
    return channels * spatial * spatial;
}
