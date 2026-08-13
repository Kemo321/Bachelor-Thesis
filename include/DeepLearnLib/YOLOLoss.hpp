#pragma once

#include "DeepLearnLib/Tensor.hpp"

/**
 * @class YOLOLoss
 * @brief Computes the loss and its derivative for YOLO-based object detection models.
 *
 * The implementation is still being migrated off LibTorch; these signatures already
 * accept dl::Tensor so Network can compile against the new tensor type.
 */
class YOLOLoss
{
public:
    /**
     * @brief Computes the YOLO loss.
     *
     * @param target Ground truth tensor. Shape: [Batch, Grid, Grid, Attributes].
     * @param prediction Predicted tensor. Shape: [Batch, Grid, Grid, Attributes].
     * @param num_classes Number of classes in the dataset. Default is 20.
     * @return Scalar tensor representing the computed loss. Shape: [1].
     */
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction, int num_classes = 20)
        -> dl::Tensor;

    /**
     * @brief Computes the derivative of the YOLO loss.
     *
     * @param target Ground truth tensor. Shape: [Batch, Grid, Grid, Attributes].
     * @param prediction Predicted tensor. Shape: [Batch, Grid, Grid, Attributes].
     * @param num_classes Number of classes in the dataset. Default is 20.
     * @return Gradient tensor. Shape: [Batch, Grid, Grid, Attributes].
     */
    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction,
                                              int num_classes = 20) -> dl::Tensor;

private:
    /**
     * @brief Calculates the Intersection over Union (IoU) between two bounding boxes.
     *
     * @param box1 First bounding box tensor. Shape: [Batch, 4].
     * @param box2 Second bounding box tensor. Shape: [Batch, 4].
     * @return IoU for each pair of bounding boxes. Shape: [Batch].
     */
    static auto calculate_iou(const dl::Tensor& box1, const dl::Tensor& box2) -> dl::Tensor;
};
