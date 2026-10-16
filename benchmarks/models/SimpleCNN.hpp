#pragma once

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/layers/Softmax.hpp"

#include <memory>
#include <vector>

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
