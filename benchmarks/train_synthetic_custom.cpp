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
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> SYNTH_CLASSES = { "square", "circle", "triangle" };

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const nlohmann::json config = load_pipeline_config("synthetic_custom");
    apply_pipeline_precision(config);
    const int batch_size = config.value("batch_size", 16);
    const int total_epochs = config.value("epochs", 800);
    const float learning_rate = config.value("learning_rate", 1.0e-4F);
    const float momentum = config.value("momentum", 0.9F);
    const float weight_decay = config.value("weight_decay", 0.0005F);
    const float gradient_clip = pipeline_gradient_clip(config);
    const int num_classes = config.value("num_classes", 3);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/Synthetic3/train"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/synthetic"));

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root.string(), train_paths, val_paths, test_paths, SYNTH_CLASSES);

    CustomDataLoader train_loader(train_paths, batch_size, true, SYNTH_CLASSES);
    CustomDataLoader test_loader(test_paths, batch_size, false, SYNTH_CLASSES);

    YOLO custom_model(num_classes);
    Network trainer(custom_model.get_all_layers(), learning_rate, gradient_clip);

    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
    }

    log_pipeline_banner({ "Synth Custom", "custom", batch_size, total_epochs, learning_rate, momentum, weight_decay,
        gradient_clip, pipeline_precision_name(config), num_classes, 0,
        static_cast<int>(custom_model.get_all_layers().size()), data_root.string(),
        static_cast<std::size_t>(train_loader.size()), static_cast<std::size_t>(test_loader.size()) });

    fs::create_directories(results_dir);
    std::ofstream csv_file((results_dir / "metrics_custom.csv").string());
    csv_file << "Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB\n";

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        auto epoch_start_time = std::chrono::steady_clock::now();
        const float current_lr = scheduled_learning_rate(config, epoch);
        apply_sgd_hyperparameters(custom_model.get_all_layers(), current_lr, momentum, weight_decay);
        for (auto& layer : custom_model.get_all_layers())
        {
            layer->train();
        }

        float epoch_train_loss = 0.0F;

        const int train_batches = for_each_prefetched_batch(train_loader,
            [&](Batch& batch, int, cudaStream_t stream)
            {
                dl::Tensor pred = custom_model.forward(batch.images, stream);
                const float batch_loss
                    = YOLOLoss::loss(batch.targets, pred, num_classes, stream).to_host(stream).front();

                dl::Tensor grad_error = trainer.clip_loss_gradient(
                    YOLOLoss::loss_derivative(batch.targets, pred, num_classes, stream));

                auto layers = custom_model.get_all_layers();
                for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
                {
                    grad_error = (*iterator)->backward(grad_error, stream);
                }
                trainer.clip_parameter_gradients(stream);
                for (auto& layer : layers)
                {
                    layer->step(stream);
                }

                epoch_train_loss += batch_loss;
            });
        float avg_train_loss = epoch_train_loss / static_cast<float>(std::max(1, train_batches));

        for (auto& layer : custom_model.get_all_layers())
        {
            layer->eval();
        }

        float epoch_test_loss = 0.0F;

        const int test_batches = for_each_prefetched_batch(test_loader,
            [&](Batch& batch, int, cudaStream_t stream)
            {
                dl::Tensor pred = custom_model.forward(batch.images, stream);
                epoch_test_loss += YOLOLoss::loss(batch.targets, pred, num_classes, stream).to_host(stream).front();
            });
        float avg_test_loss = epoch_test_loss / static_cast<float>(std::max(1, test_batches));

        auto epoch_end_time = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time).count();

        log_train_epoch({ "Synth Custom", epoch, total_epochs, current_lr, avg_train_loss, avg_test_loss, std::nullopt,
            std::nullopt, std::nullopt, train_batches, epoch_duration, current_vram_mib() });
        write_train_test_row(csv_file, epoch, avg_train_loss, avg_test_loss, epoch_duration, current_vram_mib());
    }

    std::string save_path = (results_dir / "yolov1_synthetic_custom_final.pt").string();
    trainer.save(save_path);
    log_saved("Synth Custom", save_path);
    return 0;
}
