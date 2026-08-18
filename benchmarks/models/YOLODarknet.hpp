#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * Darknet `cfg/yolov1.cfg` architecture: 24× Conv-BN-Leaky, `[local]` 3×3×256,
 * dropout, connected 7×7×(20 + 3 + 3×4) = 1715.
 *
 * This is the network that `yolov1.weights` / `extraction.conv.weights` map onto.
 * The paper-style BN-YOLO with two FC layers stays in `YOLO`.
 */
class YOLODarknet
{
public:
    static constexpr int kGridSize = 7;
    static constexpr int kBoxesPerCell = 3;
    static constexpr int kCoords = 4;
    static constexpr int kLocalFilters = 256;
    static constexpr int kLocalSpatial = 7;

    std::vector<std::shared_ptr<Layer>> backbone_layers;
    std::vector<std::shared_ptr<Layer>> head_layers;

    explicit YOLODarknet(int num_classes = 20);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;

    [[nodiscard]] auto num_classes() const -> int
    {
        return num_classes_;
    }

    [[nodiscard]] static auto prediction_size(int num_classes) -> int
    {
        return kGridSize * kGridSize * (num_classes + kBoxesPerCell + (kBoxesPerCell * kCoords));
    }

    [[nodiscard]] static auto truth_size(int num_classes) -> int
    {
        return kGridSize * kGridSize * (1 + kCoords + num_classes);
    }

    void freeze_extraction_backbone();

private:
    int num_classes_;
};
