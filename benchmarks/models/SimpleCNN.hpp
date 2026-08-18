#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Softmax.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * Compact CIFAR-style CNN assembled from generic DeepLearnLib layers.
 *
 * Lives outside the core library. Topology:
 * Conv2d -> LeakyReLU -> MaxPool2d -> Conv2d -> LeakyReLU -> MaxPool2d ->
 * Flatten -> FullyConnected -> Softmax.
 *
 * Apply CrossEntropyLoss to logits from `forward_logits`; `forward` returns
 * class probabilities after Softmax.
 */
class SimpleCNN
{
public:
    explicit SimpleCNN(int num_classes, int image_size = 32, int in_channels = 3);

    [[nodiscard]] auto forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
    [[nodiscard]] auto num_classes() const -> int;
    [[nodiscard]] auto image_size() const -> int;
    [[nodiscard]] auto in_channels() const -> int;

private:
    int num_classes_;
    int image_size_;
    int in_channels_;
    std::vector<std::shared_ptr<Layer>> layers_;
    std::shared_ptr<Softmax> softmax_;
};
