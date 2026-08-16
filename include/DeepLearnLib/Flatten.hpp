#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <vector>

/**
 * Collapses all non-batch dimensions: forward [N, ...] -> [N, F], backward restores the cached shape.
 */
class Flatten : public Layer
{
public:
    Flatten() = default;

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;

private:
    std::vector<int> input_shape_cache_;
};
