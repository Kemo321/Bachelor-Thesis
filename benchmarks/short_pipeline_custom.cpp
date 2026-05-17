#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    DataPaths train_paths, val_paths, test_paths;
    split_dataset("../../data/VOCdevkit/VOC2012", train_paths, val_paths, test_paths);

    auto loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, false).map(torch::data::transforms::Stack<>()), 
        torch::data::DataLoaderOptions().batch_size(16).workers(2));

    YOLO custom_model;
    for (auto& l : custom_model->get_all_layers()) { l->to(device); l->train(); l->learning_rate = 1e-5F; }

    std::ofstream csv("../../results/short_metrics_custom.csv");
    csv << "Epoch;Loss\n";

    for(int epoch=1; epoch<=3; ++epoch) {
        float l_sum = 0.0f;
        for(auto& batch : *loader) {
            auto d = batch.data.to(device); auto t = batch.target.to(device);
            auto pred = custom_model->forward(d);
            auto loss = YOLOLoss::loss(t, pred);
            l_sum += loss.item<float>();

            auto grad = YOLOLoss::loss_derivative(t, pred).clamp(-5.0, 5.0);
            auto layers = custom_model->get_all_layers();
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) { grad = (*it)->backward(grad); }
            for (auto& l : layers) l->step();
        }
        std::cout << "[SHORT CUSTOM] Epoch " << epoch << " Loss: " << l_sum << "\n";
        csv << epoch << ";" << l_sum << "\n";
    }
    return 0;
}