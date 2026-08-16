#pragma once

#include "DeepLearnLib/Tensor.hpp"

/**
 * YOLOv1 localization, confidence, and classification loss on the GPU.
 *
 * Predictions and targets are grids [Batch, 7, 7, 10 + num_classes]
 * (or flattened [Batch, 7*7*(10 + num_classes)]).
 */
class YOLOLoss
{
public:
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction, int num_classes = 20)
        -> dl::Tensor;

    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction,
        int num_classes = 20) -> dl::Tensor;

private:
    static auto calculate_iou(const dl::Tensor& box1, const dl::Tensor& box2) -> dl::Tensor;
};
