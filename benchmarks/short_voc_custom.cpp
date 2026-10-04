#include "experiment_config.hpp"
#include "prefetch_batch.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
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
    const std::string voc_subset = config.value("voc_subset", "VOC2012");
    const int batch_size = config.value("batch_size", 16);
    const float learning_rate = config.value("learning_rate", 1.0e-5F);
    const float momentum = config.value("momentum", 0.9F);
    const float weight_decay = config.value("weight_decay", 0.0005F);
    const float gradient_clip = pipeline_gradient_clip(config);
    const int num_classes = config.value("num_classes", 20);

    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / voc_subset).string(), train_paths, val_paths, test_paths);

    CustomDataLoader loader(train_paths, batch_size, false);
    YOLO custom_model(num_classes);
    Network trainer(custom_model.get_all_layers(), learning_rate, gradient_clip);
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->train();
    }
    apply_sgd_hyperparameters(custom_model.get_all_layers(), learning_rate, momentum, weight_decay);

    constexpr int kEpochs = 3;
    log_pipeline_banner({ "Short VOC Custom", "custom", batch_size, kEpochs, learning_rate, momentum, weight_decay,
        gradient_clip, pipeline_precision_name(config), num_classes, 0,
        static_cast<int>(custom_model.get_all_layers().size()), data_root.string(), train_paths.images.size(), 0 });

    auto csv = open_metrics_csv(results_dir, "metrics_custom.csv", "Epoch;Loss;Time(s);VRAM_MiB");
    for (int epoch = 1; epoch <= kEpochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_lr = scheduled_learning_rate(config, epoch);
        apply_sgd_hyperparameters(custom_model.get_all_layers(), current_lr, momentum, weight_decay);
        float l_sum = 0.0F;
        const int batches = for_each_prefetched_batch(loader,
            [&](Batch& batch, int, cudaStream_t stream)
            {
                dl::Tensor pred = custom_model.forward(batch.images, stream);
                l_sum += YOLOLoss::loss(batch.targets, pred, num_classes, stream).to_host(stream).front();

                dl::Tensor grad = trainer.clip_loss_gradient(
                    YOLOLoss::loss_derivative(batch.targets, pred, num_classes, stream));
                auto layers = custom_model.get_all_layers();
                for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                {
                    grad = (*it)->backward(grad, stream);
                }
                trainer.clip_parameter_gradients(stream);
                for (auto& layer : layers)
                {
                    layer->step(stream);
                }
            });
        const float avg_loss = l_sum / static_cast<float>(std::max(1, batches));
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        log_train_epoch({ "Short VOC Custom", epoch, kEpochs, current_lr, avg_loss, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, batches, elapsed, current_vram_mib() });
        write_loss_row(csv, epoch, avg_loss, elapsed, current_vram_mib());
    }
    return 0;
}
