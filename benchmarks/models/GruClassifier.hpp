#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <memory>
#include <vector>

class GruClassifier
{
public:
    GruClassifier(int input_size, int hidden_size, int num_classes);

    [[nodiscard]] auto forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream = 0) -> dl::Tensor;
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
    [[nodiscard]] auto input_size() const -> int;
    [[nodiscard]] auto hidden_size() const -> int;
    [[nodiscard]] auto num_classes() const -> int;

private:
    int input_size_;
    int hidden_size_;
    int num_classes_;
    std::vector<std::shared_ptr<Layer>> layers_;
};
