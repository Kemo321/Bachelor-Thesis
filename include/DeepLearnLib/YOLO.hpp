#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Flatten.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Dropout.hpp"

/**
 * @brief YOLO (You Only Look Once) implementation for object detection.
 * 
 * This class defines the YOLO architecture, including the backbone and head layers.
 */
struct YOLOImpl : torch::nn::Module
{
    /**
     * @brief Backbone layers of the YOLO model.
     * 
     * These layers are responsible for feature extraction from the input tensor.
     */
    std::vector<std::shared_ptr<Layer>> backbone_layers;

    /**
     * @brief Head layers of the YOLO model.
     * 
     * These layers are responsible for making predictions based on the extracted features.
     */
    std::vector<std::shared_ptr<Layer>> head_layers;

    /**
     * @brief Constructs the YOLO model.
     * 
     * @param num_classes The number of classes for object detection. Default is 20.
     */
    YOLOImpl(int num_classes = 20);

    /**
     * @brief Performs a forward pass through the YOLO model.
     * 
     * @param input_tensor Input tensor with shape [Batch, Channels, Height, Width].
     * @return torch::Tensor Output tensor with predictions. Shape depends on the model configuration.
     */
    [[nodiscard]] auto forward(torch::Tensor input_tensor) -> torch::Tensor;

    /**
     * @brief Retrieves all layers of the YOLO model.
     * 
     * @return std::vector<std::shared_ptr<Layer>> A vector containing all layers in the model.
     */
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
};

/**
 * @brief Alias for the YOLO module.
 */
TORCH_MODULE(YOLO);

/**
 * @brief Computes the gradient of the YOLO loss function manually.
 * 
 * This function calculates the gradient of the loss with respect to the predictions.
 * 
 * @param prediction The predicted tensor. Shape: [Batch, Grid, Grid, Classes + 5].
 * @param target The ground truth tensor. Shape: [Batch, Grid, Grid, Classes + 5].
 * @return torch::Tensor Gradient tensor with the same shape as the prediction tensor.
 * 
 * @note This function assumes a custom backpropagation logic for YOLOv1.
 */
[[nodiscard]] auto compute_manual_yolo_loss_gradient(const torch::Tensor& prediction, const torch::Tensor& target) -> torch::Tensor;