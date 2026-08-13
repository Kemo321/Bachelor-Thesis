#pragma once

#include <memory>
#include <vector>

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

/**
 * @brief YOLOv1 architecture implemented with custom dl::Tensor layers.
 *
 * This is a plain C++ class (not a LibTorch module). The backbone extracts
 * features; the head flattens and predicts grid-cell detections.
 */
class YOLO
{
public:
    /**
     * @brief Backbone layers of the YOLO model.
     */
    std::vector<std::shared_ptr<Layer>> backbone_layers;

    /**
     * @brief Head layers of the YOLO model.
     */
    std::vector<std::shared_ptr<Layer>> head_layers;

    /**
     * @brief Constructs the YOLO model.
     *
     * @param num_classes Number of classes for object detection. Default is 20.
     */
    explicit YOLO(int num_classes = 20);

    /**
     * @brief Performs a forward pass through the YOLO model.
     *
     * @param input_tensor Input tensor with shape [Batch, Channels, Height, Width].
     * @return Output tensor with shape [Batch, 7*7*(10 + num_classes)].
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor;

    /**
     * @brief Retrieves all layers of the YOLO model in backbone-then-head order.
     */
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
};
