#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

#include <memory>
#include <string>
#include <vector>

/**
 * Ordered stack of custom layers with a training loop and binary weight I/O.
 */
class Network
{
public:
    Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate);

    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor;
    void fit(const dl::Tensor& x_train, const dl::Tensor& y_train, int epochs, int verbose = 1);
    void save(const std::string& path);
    void load(const std::string& path);

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    YOLOLoss criterion_;
};
