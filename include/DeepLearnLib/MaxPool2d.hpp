#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>
#include <vector>

class MaxPool2d : public Layer
{
public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    MaxPool2d(int kernel_size_val, int stride_val);

    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;
    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

private:
    int kernel_size_;
    int stride_;
    torch::Tensor indices_cache_;
    std::vector<int64_t> input_shape_cache_;
};