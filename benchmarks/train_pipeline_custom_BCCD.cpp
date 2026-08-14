#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"

namespace fs = std::filesystem;

const std::vector<std::string> BCCD_CLASSES = { "RBC", "WBC", "Platelets" };

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const int batch_size = 16;
    const int total_epochs = 800;
    const std::string data_root = "../../data/BCCD_Dataset/BCCD";
    const std::string results_dir = "../../results/bccd";

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    std::cout << "[BCCD CUSTOM PIPELINE] Starting on device: " << (gpu_count > 0 ? "GPU" : "CPU") << "\n";

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root, train_paths, val_paths, test_paths, BCCD_CLASSES);

    CustomDataLoader train_loader(train_paths, batch_size, true, BCCD_CLASSES);
    CustomDataLoader test_loader(test_paths, batch_size, false, BCCD_CLASSES);

    YOLO custom_model(3);
    Network trainer(custom_model.get_all_layers(), 1e-4F);

    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
    }

    auto get_lr = [](int ep) -> float
    {
        if (ep <= 30)
            return 1e-5F;
        if (ep <= 300)
            return 5e-5F;
        if (ep <= 400)
            return 4e-5F;
        if (ep <= 800)
            return 1e-5F;
        return 1e-5F;
    };

    fs::create_directories(results_dir);
    std::ofstream csv_file(results_dir + "/metrics_custom.csv");
    csv_file << "Epoch;TrainLoss;TestLoss;Time(s)\n";

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        auto epoch_start_time = std::chrono::steady_clock::now();
        float current_lr = get_lr(epoch);

        for (auto& layer : custom_model.get_all_layers())
        {
            layer->learning_rate = current_lr;
            layer->train();
        }

        float epoch_train_loss = 0.0F;
        int train_batches = 0;

        train_loader.reset();
        while (train_loader.has_next())
        {
            Batch batch = train_loader.get_batch();

            dl::Tensor pred = custom_model.forward(batch.images);
            const float batch_loss = YOLOLoss::loss(batch.targets, pred, 3).to_host().front();

            dl::Tensor grad_error = YOLOLoss::loss_derivative(batch.targets, pred, 3);
            grad_error = grad_error.clamp(-10.0F, 10.0F);

            auto layers = custom_model.get_all_layers();
            for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
            {
                grad_error = (*iterator)->backward(grad_error);
            }
            for (auto& layer : layers)
            {
                layer->step();
            }

            epoch_train_loss += batch_loss;
            train_batches++;
        }
        float avg_train_loss = epoch_train_loss / static_cast<float>(std::max(1, train_batches));

        for (auto& layer : custom_model.get_all_layers())
        {
            layer->eval();
        }

        float epoch_test_loss = 0.0F;
        int test_batches = 0;

        test_loader.reset();
        while (test_loader.has_next())
        {
            Batch batch = test_loader.get_batch();
            dl::Tensor pred = custom_model.forward(batch.images);
            epoch_test_loss += YOLOLoss::loss(batch.targets, pred, 3).to_host().front();
            test_batches++;
        }
        float avg_test_loss = epoch_test_loss / static_cast<float>(std::max(1, test_batches));

        auto epoch_end_time = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time).count();

        std::cout << "BCCD Custom | Epoch [" << std::setw(3) << epoch << "/" << total_epochs << "] | Train Loss: "
                  << std::fixed << std::setprecision(4) << avg_train_loss << " | Test Loss: " << avg_test_loss
                  << " | Time: " << epoch_duration << "s\n";

        csv_file << epoch << ";" << avg_train_loss << ";" << avg_test_loss << ";" << epoch_duration << "\n";
        csv_file.flush();
    }

    std::string save_path = results_dir + "/yolov1_bccd_custom_final.pt";
    trainer.save(save_path);
    std::cout << "[INFO] Final model saved: " << save_path << "\n";
    return 0;
}
