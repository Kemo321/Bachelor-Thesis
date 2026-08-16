#pragma once

#include <memory>
#include <vector>

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

/**
 * YOLOv1 architecture on custom dl::Tensor layers (not a LibTorch module).
 *
 * The backbone extracts features; the head flattens and predicts the 7x7 detection grid.
 */
class YOLO
{
public:
    std::vector<std::shared_ptr<Layer>> backbone_layers;
    std::vector<std::shared_ptr<Layer>> head_layers;

    explicit YOLO(int num_classes = 20);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor;
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
};
