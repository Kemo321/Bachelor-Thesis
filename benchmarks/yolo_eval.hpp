#pragma once

#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/mAP.hpp"
#include "DeepLearnLib/utils.hpp"

#include <cstddef>
#include <iterator>
#include <vector>

inline auto slice_detection_sample(const std::vector<float>& host, int sample_index, int elements_per_sample)
    -> std::vector<float>
{
    const auto offset = static_cast<std::size_t>(sample_index) * static_cast<std::size_t>(elements_per_sample);
    return { host.begin() + static_cast<std::ptrdiff_t>(offset),
        host.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<std::size_t>(elements_per_sample)) };
}

inline auto detections_from_flat(const std::vector<float>& host, int batch, int elements_per_sample,
    float conf_threshold, int image_size, int num_classes, bool apply_suppression, float nms_threshold)
    -> std::vector<Detection>
{
    std::vector<Detection> all;
    for (int sample = 0; sample < batch; ++sample)
    {
        std::vector<float> sample_buffer = slice_detection_sample(host, sample, elements_per_sample);
        std::vector<Detection> decoded = decode_yolo_tensor(
            sample_buffer, conf_threshold, image_size, image_size, num_classes);
        if (apply_suppression)
        {
            decoded = apply_nms(decoded, nms_threshold);
        }
        all.insert(all.end(), decoded.begin(), decoded.end());
    }
    return all;
}

inline auto detections_from_tensor(const dl::Tensor& tensor, float conf_threshold, int image_size, int num_classes,
    bool apply_suppression, float nms_threshold, cudaStream_t stream = 0) -> std::vector<Detection>
{
    const std::vector<float> host = tensor.to_host(stream);
    const int batch = tensor.get_shape()[0];
    const int elements_per_sample = static_cast<int>(tensor.get_size()) / batch;
    return detections_from_flat(
        host, batch, elements_per_sample, conf_threshold, image_size, num_classes, apply_suppression, nms_threshold);
}
