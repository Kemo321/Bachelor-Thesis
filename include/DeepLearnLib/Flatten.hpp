#pragma once

#include "DeepLearnLib/Layer.hpp"
#include <torch/torch.h>
#include <vector>

class Flatten : public Layer
{
public:
    Flatten() = default;

    [[nodiscard]] auto forward(const torch::Tensor& input_tensor) -> torch::Tensor override;
    [[nodiscard]] auto backward(const torch::Tensor& output_error_derivative) -> torch::Tensor override;

private:
    std::vector<int64_t> input_shape_cache_;
};