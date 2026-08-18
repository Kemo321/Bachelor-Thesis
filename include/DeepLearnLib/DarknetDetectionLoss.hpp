#pragma once

#include "DeepLearnLib/Tensor.hpp"

/**
 * Darknet `detection_layer` loss for YOLOv1 (`cfg/yolov1.cfg`).
 *
 * Prediction layout (per image, length `S*S*(C + B + B*4)`):
 *   classes[S*S*C] | objectness[S*S*B] | boxes[S*S*B*4]
 *
 * Target layout (per image): `[S, S, 1 + coords + C]` with
 *   `[is_obj, class_onehot..., x, y, w, h]` where `x,y` are in grid units `[0, S]`
 *   and `w,h` are image-relative `[0, 1]`.
 *
 * Gradients are true dL/dpred so existing `w -= lr * g` SGD is descent.
 */
class DarknetDetectionLoss
{
public:
    struct Config
    {
        int side { 7 };
        int num_boxes { 3 };
        int coords { 4 };
        int num_classes { 20 };
        bool sqrt_wh { true };
        bool rescore { true };
        float object_scale { 1.0F };
        float noobject_scale { 0.5F };
        float class_scale { 1.0F };
        float coord_scale { 5.0F };
    };

    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction, const Config& config,
        cudaStream_t stream = 0) -> dl::Tensor;

    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction,
        const Config& config, cudaStream_t stream = 0) -> dl::Tensor;
};
