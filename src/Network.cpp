#include "DeepLearnLib/Network.hpp"
#include <iostream>
#include <utility>

Network::Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate_val)
    : layers_(std::move(layers_vector))
{
    for (const auto& layer_pointer : layers_)
    {
        layer_pointer->learning_rate = learning_rate_val;
    }
}

auto Network::forward(torch::Tensor input_tensor) -> torch::Tensor
{
    torch::Tensor current_output = std::move(input_tensor);
    for (const auto& layer_pointer : layers_)
    {
        current_output = layer_pointer->forward(current_output);
    }
    return current_output;
}

auto Network::fit(const torch::Tensor& x_train, const torch::Tensor& y_train, int epochs, int verbose) -> void
{
    torch::NoGradGuard no_grad;

    for (int epoch_idx = 0; epoch_idx < epochs; ++epoch_idx)
    {
        torch::Tensor current_prediction = forward(x_train);

        const float current_loss_value = YOLOLoss::loss(y_train, current_prediction).item<float>();

        torch::Tensor gradient_error = YOLOLoss::loss_derivative(y_train, current_prediction);

        gradient_error = gradient_error.clamp(-1.0, 1.0);

        for (auto iterator = layers_.rbegin(); iterator != layers_.rend(); ++iterator)
        {
            gradient_error = (*iterator)->backward(gradient_error);
        }

        constexpr int log_interval = 10;
        if (verbose != 0 && (epoch_idx % log_interval == 0 || epoch_idx == epochs - 1))
        {
            std::cout << "[INFO] Epoka " << epoch_idx << "/" << epochs
                      << " | Blad (Loss): " << current_loss_value << "\n";
        }
    }
}

auto Network::save(const std::string& path) -> void
{
    torch::serialize::OutputArchive archive;
    for (size_t index = 0; index < layers_.size(); ++index)
    {
        auto params = layers_[index]->get_parameters();
        for (const auto& pair : params)
        {
            archive.write("layer_" + std::to_string(index) + "_" + pair.first, pair.second);
        }
    }
    archive.save_to(path);
    std::cout << "[INFO] Model saved to: " << path << "\n";
}

auto Network::load(const std::string& path) -> void
{
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    for (size_t index = 0; index < layers_.size(); ++index)
    {
        std::map<std::string, torch::Tensor> params_to_load;
        auto current_params = layers_[index]->get_parameters();

        for (const auto& pair : current_params)
        {
            torch::Tensor tensor_to_load;
            archive.read("layer_" + std::to_string(index) + "_" + pair.first, tensor_to_load);
            params_to_load[pair.first] = tensor_to_load;
        }
        layers_[index]->set_parameters(params_to_load);
    }
    std::cout << "[INFO] Model loaded from: " << path << "\n";
}