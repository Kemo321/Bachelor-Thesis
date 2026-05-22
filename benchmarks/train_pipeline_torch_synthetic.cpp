#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <random>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

#include "DeepLearnLib/TorchYOLO.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

namespace fs = std::filesystem;

const std::vector<std::string> SYNTH_CLASSES = { "square", "circle", "triangle" };

int main() {
    std::srand(std::time(nullptr));

    const int batch_size = 16;   
    const int total_epochs = 800; 
    const std::string data_root = "../../data/Synthetic3/train";
    const std::string results_dir = "../../results/synthetic";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "[SYNTHETIC TORCH PIPELINE] Starting on device: " << (device.is_cuda() ? "GPU" : "CPU") << "\n";

    if (device.is_cuda()) {
        at::globalContext().setBenchmarkCuDNN(true);
    }

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root, train_paths, val_paths, test_paths, SYNTH_CLASSES);

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, true, SYNTH_CLASSES).map(torch::data::transforms::Stack<>()), 
        torch::data::samplers::RandomSampler(train_paths.images.size()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    auto test_loader = torch::data::make_data_loader(
        VOCYoloDataset(test_paths, false, SYNTH_CLASSES).map(torch::data::transforms::Stack<>()), 
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    YOLOv1 model(3);
    model->to(device);

    auto get_lr = [](int ep) -> float {
        if (ep <= 30) return 1e-5F;
        if (ep <= 300) return 5e-5F;
        if (ep <= 400) return 4e-5F;
        if (ep <= 800) return 1e-5F;
        return 1e-5F;
    };

    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(get_lr(1)).momentum(0.9).weight_decay(0.0005));
    
    fs::create_directories(results_dir);
    std::ofstream csv_file(results_dir + "/metrics_torch.csv");
    csv_file << "Epoch;TrainLoss;TestLoss;Time(s)\n"; 

    for (int epoch = 1; epoch <= total_epochs; ++epoch) {
        auto epoch_start_time = std::chrono::steady_clock::now(); 
        float current_lr = get_lr(epoch);
        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::SGDOptions&>(group.options()).lr(current_lr);
        }

        model->train();
        float epoch_train_loss = 0.0F;
        int train_batches = 0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device, true);   
            auto target = batch.target.to(device, true); 

            optimizer.zero_grad();
            auto pred = model->forward(data);
            auto loss = YOLOLoss::loss(target, pred, 3);
            
            loss.backward();
            optimizer.step();

            epoch_train_loss += loss.item().toFloat();
            train_batches++;
        }
        float avg_train_loss = epoch_train_loss / std::max(1, train_batches);

        model->eval();
        float epoch_test_loss = 0.0F;
        int test_batches = 0;

        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *test_loader) {
                auto data = batch.data.to(device, true);
                auto target = batch.target.to(device, true);
                auto pred = model->forward(data);
                epoch_test_loss += YOLOLoss::loss(target, pred, 3).item().toFloat();
                test_batches++;
            }
        }
        float avg_test_loss = epoch_test_loss / std::max(1, test_batches);

        auto epoch_end_time = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time).count();

        std::cout << "Synth Torch | Epoch [" << std::setw(3) << epoch << "/" << total_epochs << "] | Train Loss: " 
                  << std::fixed << std::setprecision(4) << avg_train_loss << " | Test Loss: " << avg_test_loss 
                  << " | Time: " << epoch_duration << "s\n";

        csv_file << epoch << ";" << avg_train_loss << ";" << avg_test_loss << ";" << epoch_duration << "\n";
        csv_file.flush();
    }

    std::string save_path = results_dir + "/yolov1_synthetic_torch_final.pt";
    torch::save(model, save_path);
    std::cout << "[INFO] Final model saved: " << save_path << "\n";
    return 0;
}