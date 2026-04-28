#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <torch/torch.h>

#include "DeepLearnLib/TorchYOLO.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

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
    std::cout << "[SANITY CHECK] LibTorch YOLOv1 Pipeline\n";
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

    YOLOv1 model;
    model->to(device);

    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9).weight_decay(0.0005));

    float first_epoch_loss = 0.0F;
    float last_epoch_loss = 0.0F;

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        std::cout << "\n--- EPOCH " << epoch << " / " << total_epochs << " ---\n";
        model->train();
        float epoch_train_loss = 0.0F;
        int processed_epoch = 0;

        for (auto& batch : *train_loader)
        {
            auto data = batch.data.to(device, true);
            auto target = batch.target.to(device, true);

            optimizer.zero_grad();
            auto pred = model->forward(data);
            auto loss = YOLOLoss::loss(target, pred);

            loss.backward();
            optimizer.step();

            if (device.is_cuda())
            {
                torch::cuda::synchronize();
            }

            float batch_loss = loss.item<float>();
            epoch_train_loss += batch_loss;
            processed_epoch += data.size(0);

            if (processed_epoch % (batch_size * 10) == 0)
            {
                std::cout << "[TRAIN] Processed: " << std::setw(5) << processed_epoch
                          << " | Loss: " << std::fixed << std::setprecision(4) << batch_loss << "\n";
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