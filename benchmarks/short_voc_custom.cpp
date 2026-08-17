#include "experiment_config.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Precision.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "YOLO.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>

int main()
{
    const nlohmann::json config = load_pipeline_config("voc_custom");
    apply_pipeline_precision(config);
    const auto data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const auto results_dir = resolve_from_source("results/voc_short");

    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / "VOC2012").string(), train_paths, val_paths, test_paths);

    CustomDataLoader loader(train_paths, 16, false);
    YOLO custom_model;
    Network trainer(custom_model.get_all_layers(), 1e-5F);
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->train();
        layer->learning_rate = 1e-5F;
    }

    auto csv = open_metrics_csv(results_dir, "metrics_custom.csv", "Epoch;Loss;Time(s);VRAM_MiB");
    constexpr int kEpochs = 3;
    for (int epoch = 1; epoch <= kEpochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        float l_sum = 0.0F;
        int batches = 0;
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
            ++batches;
        }
        const float avg_loss = l_sum / static_cast<float>(std::max(1, batches));
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        const auto vram = current_vram_mib();
        log_train_epoch("Short VOC Custom", epoch, kEpochs, avg_loss, elapsed, vram);
        write_loss_row(csv, epoch, avg_loss, elapsed, vram);
    }
    return 0;
}
