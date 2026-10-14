#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * CIFAR-style CNN whose convolution blocks are fused Conv-BatchNorm-LeakyReLU.
 *
 * FusedCBR2d -> MaxPool2d -> FusedCBR2d -> MaxPool2d -> Flatten -> FullyConnected.
 * The Torch twin is TorchFusedCbrCNN: Conv2d, BatchNorm2d, LeakyReLU(0.1), then the same pool and head.
 * Epsilon is 1e-5 and the batch-norm blend of the new batch is 0.1, matching both stacks.
 */
class FusedCbrCNN
{
public:
    explicit FusedCbrCNN(int num_classes, int image_size = 32, int in_channels = 3);

    [[nodiscard]] auto forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
    [[nodiscard]] auto num_classes() const -> int;
    [[nodiscard]] auto image_size() const -> int;
    [[nodiscard]] auto in_channels() const -> int;

private:
    int num_classes_;
    int image_size_;
    int in_channels_;
    std::vector<std::shared_ptr<Layer>> layers_;
};
