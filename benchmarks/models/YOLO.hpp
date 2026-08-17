#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * YOLOv1 architecture assembled from generic DeepLearnLib layers.
 *
 * This type lives outside the core library: DeepLearnLib provides Tensor, Layer,
 * and kernels; applications compose them into networks. The backbone extracts
 * features with fused Conv-BN-LeakyReLU blocks; the head flattens and predicts
 * the 7x7 detection grid.
 */
class YOLO
{
public:
    std::vector<std::shared_ptr<Layer>> backbone_layers;
    std::vector<std::shared_ptr<Layer>> head_layers;

    explicit YOLO(int num_classes = 20);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
};
