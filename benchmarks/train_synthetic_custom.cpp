#include "experiment_config.hpp"
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
#include <cuda_runtime.h>
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

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    LOG_INFO("[SYNTHETIC CUSTOM PIPELINE] Starting on device: {}", gpu_count > 0 ? "GPU" : "CPU");
    LOG_INFO("[CONFIG] batch_size={} epochs={} learning_rate={} momentum={} weight_decay={} gradient_clip={} dataset_root={}",
        batch_size, total_epochs, learning_rate, momentum, weight_decay, gradient_clip, data_root.string());

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
        int train_batches = 0;

        train_loader.reset();
        while (train_loader.has_next())
        {
            Batch batch = train_loader.get_batch();

            dl::Tensor pred = custom_model.forward(batch.images);
            const float batch_loss = YOLOLoss::loss(batch.targets, pred, num_classes).to_host().front();

            dl::Tensor grad_error = trainer.clip_loss_gradient(YOLOLoss::loss_derivative(batch.targets, pred, num_classes));

            auto layers = custom_model.get_all_layers();
            for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
            {
                grad_error = (*iterator)->backward(grad_error);
            }
            trainer.clip_parameter_gradients();
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
            epoch_test_loss += YOLOLoss::loss(batch.targets, pred, num_classes).to_host().front();
            test_batches++;
        }
        float avg_test_loss = epoch_test_loss / static_cast<float>(std::max(1, test_batches));

        auto epoch_end_time = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time).count();

        const auto vram = current_vram_mib();
        log_train_epoch("Synth Custom", epoch, total_epochs, avg_train_loss, avg_test_loss, epoch_duration, vram);
        write_train_test_row(csv_file, epoch, avg_train_loss, avg_test_loss, epoch_duration, vram);
    }

    std::string save_path = (results_dir / "yolov1_synthetic_custom_final.pt").string();
    trainer.save(save_path);
    LOG_INFO("Final model saved: {}", save_path);
    return 0;
}
