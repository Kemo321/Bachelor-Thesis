#include "DeepLearnLib/Network.hpp"
#include <iostream>
#include <utility>

Network::Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate_val)
    : layers_(std::move(layers_vector))
{
    for (const auto& layer_ptr : layers_)
    {
        layer_ptr->learning_rate = learning_rate_val;
    }
}

auto Network::forward(torch::Tensor input_tensor) -> torch::Tensor
{
    torch::Tensor current_output = std::move(input_tensor);
    for (const auto& layer_ptr : layers_)
    {
        current_output = layer_ptr->forward(current_output);
    }
    return current_output;
}

void Network::fit(const torch::Tensor& x_train, const torch::Tensor& y_train, int epochs, int verbose)
{
    torch::NoGradGuard no_grad;

    for (int epoch_idx = 0; epoch_idx < epochs; ++epoch_idx)
    {
        torch::Tensor current_prediction = forward(x_train);

        const float current_loss_value = YOLOLoss::loss(y_train, current_prediction);

        torch::Tensor gradient_error = YOLOLoss::loss_derivative(y_train, current_prediction);

        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        {
            gradient_error = (*it)->backward(gradient_error);
        }

        constexpr int log_interval = 10;
        if (verbose != 0 && (epoch_idx % log_interval == 0 || epoch_idx == epochs - 1))
        {
            std::cout << "[INFO] Epoka " << epoch_idx << "/" << epochs
                      << " | Blad (Loss): " << current_loss_value << "\n";
        }
    }
}
