#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include "DeepLearnLib/TorchYOLO.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    DataPaths train_paths, val_paths, test_paths;
    split_dataset("../../data/VOCdevkit/VOC2012", train_paths, val_paths, test_paths);

    auto loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, false).map(torch::data::transforms::Stack<>()), 
        torch::data::DataLoaderOptions().batch_size(16).workers(2));

    YOLOv1 model; model->to(device); model->train();
    torch::optim::SGD opt(model->parameters(), 1e-5);

    std::ofstream csv("../../results/short_metrics_torch.csv");
    csv << "Epoch;Loss\n";

    for(int epoch=1; epoch<=3; ++epoch) {
        float l_sum = 0.0f;
        for(auto& batch : *loader) {
            auto d = batch.data.to(device); auto t = batch.target.to(device);
            opt.zero_grad();
            auto pred = model->forward(d);
            auto loss = YOLOLoss::loss(t, pred);
            loss.backward(); opt.step();
            l_sum += loss.item<float>();
        }
        std::cout << "[SHORT TORCH] Epoch " << epoch << " Loss: " << l_sum << "\n";
        csv << epoch << ";" << l_sum << "\n";
    }
    return 0;
}