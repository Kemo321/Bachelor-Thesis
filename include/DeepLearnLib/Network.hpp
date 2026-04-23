#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

class Network
{
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate_val);

    [[nodiscard]] auto forward(torch::Tensor input_tensor) -> torch::Tensor;

    void fit(const torch::Tensor& x_train, const torch::Tensor& y_train, int epochs, int verbose = 1);

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    YOLOLoss criterion_;
};
