#pragma once

#include "DeepLearnLib/core/Tensor.hpp"

class MSELoss
{
public:
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
};

class CrossEntropyLoss
{
public:
    [[nodiscard]] static auto loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
    [[nodiscard]] static auto loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor;
};
