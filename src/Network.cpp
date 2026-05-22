#include "DeepLearnLib/Network.hpp"
#include <iostream>
#include <utility>

/**
 * @brief Constructs a neural network from an ordered list of layers.
 *
 * The same learning rate is assigned to every layer to keep optimization
 * behavior consistent across the model.
 *
 * @param layers_vector Ordered layer sequence that defines the forward pass.
 * @param learning_rate_val Learning rate used by each layer during parameter updates.
 */
Network::Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate_val)
    : layers_(std::move(layers_vector))
{
    for (const auto& layer_pointer : layers_)
    {
        layer_pointer->learning_rate = learning_rate_val;
    }
}

/**
 * @brief Executes the forward pass through all layers.
 *
 * @param input_tensor Input tensor with shape compatible with the first layer,
 *        for example [Batch, Features] or [Batch, Channels, H, W].
 * @return torch::Tensor Output prediction tensor produced by the last layer.
 */
auto Network::forward(torch::Tensor input_tensor) -> torch::Tensor
{
    torch::Tensor current_output = std::move(input_tensor);
    for (const auto& layer_pointer : layers_)
    {
        current_output = layer_pointer->forward(current_output);
    }
    return current_output;
}

/**
 * @brief Trains the network for a fixed number of epochs.
 *
 * The method preserves the original learning flow: prediction, loss evaluation,
 * gradient computation, gradient clamping for stability, reverse-order backward
 * propagation using the chain rule, and parameter updates.
 *
 * @param x_train Training input tensor with shape [Batch, ...].
 * @param y_train Target tensor aligned with the model output shape, for example [Batch, ...].
 * @param epochs Number of optimization steps to run.
 * @param verbose Non-zero value enables periodic training loss logging.
 */
auto Network::fit(const torch::Tensor& x_train, const torch::Tensor& y_train, int epochs, int verbose) -> void
{
    torch::NoGradGuard no_grad;

    for (int epoch_idx = 0; epoch_idx < epochs; ++epoch_idx)
    {
        torch::Tensor prediction = forward(x_train);

        const float loss_value = YOLOLoss::loss(y_train, prediction).item<float>();

        torch::Tensor gradient_error = YOLOLoss::loss_derivative(y_train, prediction);

        gradient_error = gradient_error.clamp(-5.0F, 5.0F);

        for (auto iterator = layers_.rbegin(); iterator != layers_.rend(); ++iterator)
        {
            // Backpropagate gradients through each layer in reverse order.
            gradient_error = (*iterator)->backward(gradient_error);
        }

        for (auto& layer : layers_)
        {
            layer->step();
        }

        constexpr int log_interval = 10;
        if (verbose != 0 && (epoch_idx % log_interval == 0 || epoch_idx == epochs - 1))
        {
            std::cout << "[INFO] Epoka " << epoch_idx << "/" << epochs
                      << " | Blad (Loss): " << loss_value << "\n";
        }
    }
}

/**
 * @brief Saves all layer parameters to disk.
 *
 * Each tensor is stored under a deterministic key so the exact model state can
 * be reconstructed during loading.
 *
 * @param path Destination file path for the serialized model.
 */
auto Network::save(const std::string& path) -> void
{
    torch::serialize::OutputArchive archive;
    for (size_t index = 0; index < layers_.size(); ++index)
    {
        auto parameters = layers_[index]->get_parameters();
        for (const auto& pair : parameters)
        {
            archive.write("layer_" + std::to_string(index) + "_" + pair.first, pair.second);
        }
    }
    archive.save_to(path);
    std::cout << "[INFO] Model saved to: " << path << "\n";
}

/**
 * @brief Loads all layer parameters from disk.
 *
 * The serialized keys must match the architecture used during saving. Each
 * tensor is read back and assigned to the corresponding layer to preserve the
 * trained state exactly.
 *
 * @param path Source file path for the serialized model.
 */
auto Network::load(const std::string& path) -> void
{
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    for (size_t index = 0; index < layers_.size(); ++index)
    {
        std::map<std::string, torch::Tensor> parameters_to_load;
        auto current_parameters = layers_[index]->get_parameters();

        for (const auto& pair : current_parameters)
        {
            torch::Tensor tensor_to_load;
            archive.read("layer_" + std::to_string(index) + "_" + pair.first, tensor_to_load);
            parameters_to_load[pair.first] = tensor_to_load;
        }
        layers_[index]->set_parameters(parameters_to_load);
    }
    std::cout << "[INFO] Model loaded from: " << path << "\n";
}
