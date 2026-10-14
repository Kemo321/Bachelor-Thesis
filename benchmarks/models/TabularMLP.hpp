#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <memory>
#include <vector>

/**
 * Two-layer MLP for a table of numeric features.
 *
 * FullyConnected -> LeakyReLU(0.1) -> FullyConnected. The Torch twin is
 * TorchTabularMLP. Cross-entropy is applied to logits from `forward_logits`.
 */
class TabularMLP
{
public:
    TabularMLP(int in_features, int hidden_size, int num_classes);

    [[nodiscard]] auto forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
    [[nodiscard]] auto in_features() const -> int;
    [[nodiscard]] auto hidden_size() const -> int;
    [[nodiscard]] auto num_classes() const -> int;

private:
    int in_features_;
    int hidden_size_;
    int num_classes_;
    std::vector<std::shared_ptr<Layer>> layers_;
};
