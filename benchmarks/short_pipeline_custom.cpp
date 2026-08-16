#include <fstream>

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Precision.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"

int main()
{
    DataPaths train_paths, val_paths, test_paths;
    split_dataset("../../data/VOCdevkit/VOC2012", train_paths, val_paths, test_paths);

    CustomDataLoader loader(train_paths, 16, false);

    YOLO custom_model;
    Network trainer(custom_model.get_all_layers(), 1e-5F);
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

            dl::Tensor grad = trainer.clip_loss_gradient(YOLOLoss::loss_derivative(batch.targets, pred));
            auto layers = custom_model.get_all_layers();
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {
                grad = (*it)->backward(grad);
            }
            trainer.clip_parameter_gradients();
            for (auto& layer : layers)
            {
                layer->step();
            }
        }
        LOG_INFO("[SHORT CUSTOM] Epoch {} Loss: {}", epoch, l_sum);
        csv << epoch << ";" << l_sum << "\n";
    }
    return 0;
}
