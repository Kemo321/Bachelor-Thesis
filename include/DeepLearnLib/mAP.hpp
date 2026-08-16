#pragma once

#include <vector>

/**
 * @brief Axis-aligned detection used by mAP and (when OpenCV is present) NMS/decode.
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

/**
 * @brief Intersection-over-union of two axis-aligned Detection boxes.
 */
[[nodiscard]] auto detection_iou(const Detection& predicted, const Detection& ground_truth) -> float;

/**
 * @brief VOC-style 11-point mean Average Precision at a fixed IoU threshold (default 0.5).
 *
 * Predictions are matched greedily per class (highest score first). Each ground-truth
 * box is consumed at most once. Classes that appear in neither set are skipped.
 * The returned value is the unweighted mean of per-class AP.
 */
[[nodiscard]] auto mean_average_precision(const std::vector<Detection>& predicted,
    const std::vector<Detection>& ground_truth, float iou_threshold = 0.5F) -> float;
