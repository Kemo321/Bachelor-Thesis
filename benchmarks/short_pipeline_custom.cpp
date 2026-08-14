#include <fstream>
#include <iostream>

#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"

int main()
{
    DataPaths train_paths, val_paths, test_paths;
    split_dataset("../../data/VOCdevkit/VOC2012", train_paths, val_paths, test_paths);

    CustomDataLoader loader(train_paths, 16, false);

    YOLO custom_model;
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->train();
        layer->learning_rate = 1e-5F;
    }

    std::ofstream csv("../../results/short_metrics_custom.csv");
    csv << "Epoch;Loss\n";

    for (int epoch = 1; epoch <= 3; ++epoch)
    {
        float l_sum = 0.0f;
        loader.reset();
        while (loader.has_next())
        {
            Batch batch = loader.get_batch();
            dl::Tensor pred = custom_model.forward(batch.images);
            l_sum += YOLOLoss::loss(batch.targets, pred).to_host().front();

            dl::Tensor grad = YOLOLoss::loss_derivative(batch.targets, pred).clamp(-5.0F, 5.0F);
            auto layers = custom_model.get_all_layers();
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {
                grad = (*it)->backward(grad);
            }
            for (auto& layer : layers)
            {
                layer->step();
            }
        }
        std::cout << "[SHORT CUSTOM] Epoch " << epoch << " Loss: " << l_sum << "\n";
        csv << epoch << ";" << l_sum << "\n";
    }
    return 0;
}
