#pragma once

#include <vector>

/**
 * Axis-aligned detection used by mAP and (when OpenCV is present) NMS/decode.
 *
 * Box coordinates are top-left origin with width/height in the same units as IoU.
 */
struct Detection
{
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
    float score = 0.0F;
    int class_id = 0;
};

[[nodiscard]] auto detection_iou(const Detection& predicted, const Detection& ground_truth) -> float;

/**
 * VOC-style 11-point mean Average Precision at a fixed IoU threshold (default 0.5).
 *
 * Predictions are matched greedily per class (highest score first). Each ground-truth
 * box is consumed at most once. The returned value is the unweighted mean of per-class AP.
 */
[[nodiscard]] auto mean_average_precision(const std::vector<Detection>& predicted,
    const std::vector<Detection>& ground_truth, float iou_threshold = 0.5F) -> float;
