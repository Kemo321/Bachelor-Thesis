#pragma once

#include "DeepLearnLib/Tensor.hpp"

/**
 * @brief YOLOv1 localization, confidence, and classification loss on the GPU.
 *
 * Predictions and targets are NCHW-style grids [Batch, 7, 7, 10 + num_classes]
 * (or the equivalent flattened [Batch, 7*7*(10 + num_classes)] layout).
 */
class YOLOLoss
{
public:
    /**
     * @brief Computes the mean YOLOv1 loss.
     *
     * @param target Ground truth tensor.
     * @param prediction Predicted tensor.
     * @param num_classes Number of classes. Default is 20.
     * @return Scalar GPU tensor of shape [1].
     */
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction, int num_classes = 20)
        -> dl::Tensor;

    /**
     * @brief Computes dL/d(prediction) for the YOLOv1 loss.
     *
     * @param target Ground truth tensor.
     * @param prediction Predicted tensor.
     * @param num_classes Number of classes. Default is 20.
     * @return Gradient with the same layout as prediction.
     */
    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction,
                                              int num_classes = 20) -> dl::Tensor;

private:
    /**
     * @brief Pairwise IoU for packed [N, 4] boxes in [cx, cy, w, h] format.
     */
    static auto calculate_iou(const dl::Tensor& box1, const dl::Tensor& box2) -> dl::Tensor;
};
