#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <optional>

/**
 * @brief LeakyReLU activation implemented with Thrust on the GPU.
 */
class LeakyReLU : public Layer
{
public:
    /**
     * @brief Constructs a LeakyReLU layer.
     * @param slope_val Slope applied to negative inputs. Default is 0.1F.
     */
    explicit LeakyReLU(float slope_val = 0.1F);

    /**
     * @brief Applies LeakyReLU element-wise.
     * @param input_tensor Input activations of arbitrary shape.
     * @return Activated tensor with the same shape.
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor override;

    /**
     * @brief Backpropagates through LeakyReLU using the cached input.
     * @param output_error_derivative Upstream gradient, same shape as the forward input.
     * @return Gradient with respect to the input.
     */
    [[nodiscard]] auto backward(const dl::Tensor& output_error_derivative) -> dl::Tensor override;

private:
    std::optional<dl::Tensor> input_cache_;
    float slope_;
};
