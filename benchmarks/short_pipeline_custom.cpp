#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <torch/torch.h>

#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/dataset.hpp"

auto get_current_time_sec() -> double
{
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

auto main(int argc, char* argv[]) -> int
{
    const int batch_size = 16;
    const int total_epochs = 3;
    const std::string data_root = "../data/VOCdevkit";
    const float learning_rate = 1e-6F;

    std::cout << "========================================\n";
    std::cout << "[SANITY CHECK] Custom YOLO Pipeline\n";
    std::cout << "========================================\n";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root + "/VOC2012", train_paths, val_paths, test_paths);

    if (train_paths.images.empty())
    {
        std::cerr << "[ERROR] Missing training data!\n";
        return -1;
    }

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, true).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    YOLO custom_model;
    
    for (const auto& layer_pointer : custom_model->get_all_layers())
    {
        layer_pointer->to(device);
        layer_pointer->learning_rate = learning_rate;
    }

    Network trainer(custom_model->get_all_layers(), learning_rate);

    float first_epoch_loss = 0.0F;
    float last_epoch_loss = 0.0F;

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        std::cout << "\n--- EPOCH " << epoch << " / " << total_epochs << " ---\n";
        float epoch_train_loss = 0.0F;
        int processed_epoch = 0;

        double log_start_time = get_current_time_sec();

        {
            torch::NoGradGuard no_grad;

            for (auto& batch : *train_loader)
            {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);

                auto pred = custom_model->forward(data);

                auto batch_losss = YOLOLoss::loss(target, pred);
                float batch_loss = YOLOLoss::loss(target, pred).item<float>();
                auto grad_error = YOLOLoss::loss_derivative(target, pred);

                grad_error = grad_error.clamp(-1.0, 1.0);

                auto layers = custom_model->get_all_layers();
                for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
                {
                    grad_error = (*iterator)->backward(grad_error);
                }

                if (device.is_cuda())
                {
                    torch::cuda::synchronize();
                }

                epoch_train_loss += batch_loss;
                processed_epoch += data.size(0);

                if (processed_epoch % (batch_size * 10) == 0)
                {
                    std::cout << "[TRAIN] Processed: " << std::setw(5) << processed_epoch
                              << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss << "\n";
                }
            }
        }

        float avg_train_loss = epoch_train_loss / static_cast<float>(train_paths.images.size() / batch_size + 1);

        std::cout << "[EPOCH SUMMARY " << epoch << "] Average Training Loss: " << avg_train_loss << "\n";

        if (epoch == 1) first_epoch_loss = avg_train_loss;
        if (epoch == total_epochs) last_epoch_loss = avg_train_loss;
    }

    std::cout << "========================================\n";
    std::cout << "[SANITY CHECK RESULTS]\n";
    std::cout << "Epoch 1 Loss: " << first_epoch_loss << "\n";
    std::cout << "Epoch 3 Loss: " << last_epoch_loss << "\n";
    
    if (last_epoch_loss < first_epoch_loss)
    {
        std::cout << "STATUS: PASS (The model is learning, loss is decreasing!)\n";
    }
    else
    {
        std::cout << "STATUS: FAIL (Loss did not decrease. Check gradients or data.)\n";
    }
    std::cout << "========================================\n";

    return 0;
}