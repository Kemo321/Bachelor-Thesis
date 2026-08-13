#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

#include <memory>
#include <string>
#include <vector>

/**
 * @brief Neural network composed of ordered custom layers.
 *
 * Provides forward propagation, a training loop, and binary save/load of weights.
 */
class Network
{
public:
    /**
     * @brief Constructs a Network object.
     *
     * @param layers_vector Ordered layers that define the forward pass.
     * @param learning_rate Learning rate assigned to every layer.
     */
    Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate);

    /**
     * @brief Performs forward propagation through the network.
     *
     * @param input_tensor Input tensor with shape compatible with the first layer.
     * @return Output tensor produced by the last layer.
     */
    [[nodiscard]] auto forward(const dl::Tensor& input_tensor) -> dl::Tensor;

    /**
     * @brief Trains the network on the provided data.
     *
     * @param x_train Input training tensor.
     * @param y_train Ground-truth tensor aligned with the model output.
     * @param epochs Number of training epochs.
     * @param verbose Non-zero value enables periodic loss logging.
     */
    void fit(const dl::Tensor& x_train, const dl::Tensor& y_train, int epochs, int verbose = 1);

    /**
     * @brief Saves all layer parameters to a custom binary file.
     */
    void save(const std::string& path);

    /**
     * @brief Loads all layer parameters from a custom binary file.
     */
    void load(const std::string& path);

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    YOLOLoss criterion_;
};
