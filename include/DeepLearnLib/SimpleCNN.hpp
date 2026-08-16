#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Softmax.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * Compact CIFAR-style CNN on custom dl::Tensor layers.
 *
 * Conv2d -> LeakyReLU -> MaxPool2d -> Conv2d -> LeakyReLU -> MaxPool2d ->
 * Flatten -> FullyConnected -> Softmax. CrossEntropyLoss should be applied to
 * logits (forward_logits); Softmax is used for probabilities / accuracy.
 */
class SimpleCNN
{
public:
    explicit SimpleCNN(int num_classes, int image_size = 32);

    [[nodiscard]] auto forward_logits(const dl::Tensor& input_tensor) -> dl::Tensor;
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor;
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
    [[nodiscard]] auto num_classes() const -> int;
    [[nodiscard]] auto image_size() const -> int;

private:
    int num_classes_;
    int image_size_;
    std::vector<std::shared_ptr<Layer>> layers_;
    std::shared_ptr<Softmax> softmax_;
};
