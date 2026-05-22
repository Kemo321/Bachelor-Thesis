#pragma once
#include <torch/torch.h>

/**
 * @class YOLOLoss
 * @brief Computes the loss and its derivative for YOLO-based object detection models.
 * 
 * This class provides static methods to calculate the loss and its derivative for YOLO models.
 * It also includes a utility function to compute the Intersection over Union (IoU) between bounding boxes.
 */
class YOLOLoss {
public:
    /**
     * @brief Computes the YOLO loss.
     * 
     * @param target The ground truth tensor. Shape: [Batch, Grid, Grid, Attributes].
     * @param prediction The predicted tensor. Shape: [Batch, Grid, Grid, Attributes].
     * @param numClasses The number of classes in the dataset. Default is 20.
     * @return A tensor representing the computed loss. Shape: [1].
     */
    [[nodiscard]] static auto loss(const torch::Tensor& target, const torch::Tensor& prediction, int numClasses = 20) -> torch::Tensor;

    /**
     * @brief Computes the derivative of the YOLO loss.
     * 
     * @param target The ground truth tensor. Shape: [Batch, Grid, Grid, Attributes].
     * @param prediction The predicted tensor. Shape: [Batch, Grid, Grid, Attributes].
     * @param numClasses The number of classes in the dataset. Default is 20.
     * @return A tensor representing the derivative of the loss. Shape: [Batch, Grid, Grid, Attributes].
     */
    [[nodiscard]] static auto loss_derivative(const torch::Tensor& target, const torch::Tensor& prediction, int numClasses = 20) -> torch::Tensor;

private:
    /**
     * @brief Calculates the Intersection over Union (IoU) between two bounding boxes.
     * 
     * @param box1 The first bounding box tensor. Shape: [Batch, 4].
     * @param box2 The second bounding box tensor. Shape: [Batch, 4].
     * @return A tensor representing the IoU for each pair of bounding boxes. Shape: [Batch].
     */
    static auto calculate_iou(const torch::Tensor& box1, const torch::Tensor& box2) -> torch::Tensor;
};
