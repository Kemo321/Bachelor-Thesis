#pragma once

#include <torch/torch.h>

/**
 * @brief YOLOv1 implementation using LibTorch.
 * 
 * This class defines the YOLOv1 model architecture, including a backbone for feature extraction
 * and a head for bounding box and class prediction.
 */
struct YOLOv1Impl : torch::nn::Module
{
    /**
     * @brief Constructs the YOLOv1 model.
     * 
     * @param num_classes Number of classes for object detection. Default is 20.
     */
    YOLOv1Impl(int num_classes = 20);

    /**
     * @brief Performs a forward pass through the YOLOv1 model.
     * 
     * @param input_tensor Input tensor with shape [Batch, Channels, Height, Width].
     * @return torch::Tensor Output tensor with predictions, shape [Batch, S, S, B * 5 + num_classes].
     */
    [[nodiscard]] auto forward(torch::Tensor input_tensor) -> torch::Tensor;

private:
    int num_classes_; ///< Number of classes for object detection.

    torch::nn::Sequential backbone{nullptr}; ///< Backbone network for feature extraction.
    torch::nn::Sequential head{nullptr}; ///< Head network for bounding box and class prediction.
};

/**
 * @brief Alias for the YOLOv1 module.
 */
TORCH_MODULE(YOLOv1);

/**
 * @brief Computes the YOLOv1 loss function.
 * 
 * This function calculates the loss for the YOLOv1 model, which includes localization loss,
 * confidence loss, and classification loss.
 * 
 * @param prediction Predicted tensor from the model, shape [Batch, S, S, B * 5 + num_classes].
 * @param target Ground truth tensor, shape [Batch, S, S, B * 5 + num_classes].
 * @return torch::Tensor Scalar tensor representing the total loss.
 */
[[nodiscard]] auto compute_yolo_loss(const torch::Tensor& prediction, const torch::Tensor& target) -> torch::Tensor;