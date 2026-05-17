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

#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

namespace fs = std::filesystem;

const std::vector<std::string> SYNTH_CLASSES = { "square", "circle", "triangle" };

int main() {
    std::srand(std::time(nullptr));

    const int batch_size = 16;   
    const int total_epochs = 150; 
    const std::string data_root = "../../data/Synthetic3/train";
    const std::string results_dir = "../../results/synthetic";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "[SYNTHETIC CUSTOM PIPELINE] Start na urzadzeniu: " << (device.is_cuda() ? "GPU" : "CPU") << "\n";

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

    YOLO custom_model(3); 
    Network trainer(custom_model->get_all_layers(), 1e-4F);

    for (auto& layer : custom_model->get_all_layers()) {
        layer->to(device);
    }

    auto get_lr = [](int ep) -> float {
        if (ep <= 20) return 2e-5F;
        if (ep <= 100) return 1e-4F;
        return 2e-5F;
    };

    fs::create_directories(results_dir);
    std::ofstream csv_file(results_dir + "/metrics_custom.csv");
    csv_file << "Epoch;TrainLoss;TestLoss;Time(s)\n"; 

    for (int epoch = 1; epoch <= total_epochs; ++epoch) {
        auto epoch_start_time = std::chrono::steady_clock::now(); 
        float current_lr = get_lr(epoch);
        
        for (auto& layer : custom_model->get_all_layers()) {
            layer->learning_rate = current_lr;
            layer->train(); 
        }

        float epoch_train_loss = 0.0F;
        int train_batches = 0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device, true);
            auto target = batch.target.to(device, true);

            auto pred = custom_model->forward(data);
            float batch_loss = YOLOLoss::loss(target, pred, 3).item().toFloat();
            
            auto grad_error = YOLOLoss::loss_derivative(target, pred, 3);
            grad_error = grad_error.clamp(-10.0, 10.0);

            auto layers = custom_model->get_all_layers();
            for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator) {
                grad_error = (*iterator)->backward(grad_error);
            }
            for (auto& layer : layers) {
                layer->step();
            }

            epoch_train_loss += batch_loss;
            train_batches++;
        }
        float avg_train_loss = epoch_train_loss / std::max(1, train_batches);

        for (auto& layer : custom_model->get_all_layers()) {
            layer->eval(); 
        }

        float epoch_test_loss = 0.0F;
        int test_batches = 0;

        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *test_loader) {
                auto data = batch.data.to(device, true);
                auto target = batch.target.to(device, true);
                auto pred = custom_model->forward(data);
                
                epoch_test_loss += YOLOLoss::loss(target, pred, 3).item().toFloat();
                test_batches++;
            }
        }
        float avg_test_loss = epoch_test_loss / std::max(1, test_batches);

        auto epoch_end_time = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time).count();

        std::cout << "Synth Custom | Epoka [" << std::setw(3) << epoch << "/" << total_epochs << "] | Train Loss: " 
                  << std::fixed << std::setprecision(4) << avg_train_loss << " | Test Loss: " << avg_test_loss 
                  << " | Czas: " << epoch_duration << "s\n";

        csv_file << epoch << ";" << avg_train_loss << ";" << avg_test_loss << ";" << epoch_duration << "\n";
        csv_file.flush();
    }

    std::string save_path = results_dir + "/yolov1_synthetic_custom_final.pt";
    trainer.save(save_path);
    std::cout << "[INFO] Zapisano ostateczny model: " << save_path << "\n";
    return 0;
}