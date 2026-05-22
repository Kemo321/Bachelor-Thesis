#pragma once

#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

/**
 * @brief Represents a neural network composed of multiple layers.
 * 
 * This class provides functionality for forward propagation, training (fitting),
 * and saving/loading the model. It uses YOLOLoss as the loss criterion.
 */
class Network
{
public:
    /**
     * @brief Constructs a Network object.
     * 
     * @param layersVector A vector of shared pointers to Layer objects representing the network's layers.
     * @param learningRate The learning rate for the optimizer.
     */
    Network(std::vector<std::shared_ptr<Layer>> layersVector, float learningRate);

    /**
     * @brief Performs forward propagation through the network.
     * 
     * @param inputTensor The input tensor with shape [Batch, Channels, Height, Width].
     * @return torch::Tensor The output tensor after forward propagation.
     */
    [[nodiscard]] auto forward(torch::Tensor inputTensor) -> torch::Tensor;

    /**
     * @brief Trains the network on the provided data.
     * 
     * @param xTrain The input training data tensor with shape [Batch, Channels, Height, Width].
     * @param yTrain The ground truth tensor with shape [Batch, ...] (depends on the task).
     * @param epochs The number of training epochs.
     * @param verbose The verbosity level (default is 1).
     */
    void fit(const torch::Tensor& xTrain, const torch::Tensor& yTrain, int epochs, int verbose = 1);

    /**
     * @brief Saves the network's parameters to a file.
     * 
     * @param path The file path where the model will be saved.
     */
    void save(const std::string& path);

    /**
     * @brief Loads the network's parameters from a file.
     * 
     * @param path The file path from which the model will be loaded.
     */
    void load(const std::string& path);

private:
    std::vector<std::shared_ptr<Layer>> layers_; ///< The layers of the network.
    YOLOLoss criterion_; ///< The loss function used for training.
};
