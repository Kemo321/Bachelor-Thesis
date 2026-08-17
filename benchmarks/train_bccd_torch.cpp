#include "experiment_config.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "TorchDataset.hpp"
#include "TorchYOLO.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> BCCD_CLASSES = { "RBC", "WBC", "Platelets" };

int main()
{
    std::srand(std::time(nullptr));

    const nlohmann::json config = load_pipeline_config("bccd_torch");
    const int batch_size = config.value("batch_size", 16);
    const int total_epochs = config.value("epochs", 800);
    const int num_classes = config.value("num_classes", 3);
    const int dataloader_workers = config.value("dataloader_workers", 4);
    const double momentum = config.value("momentum", 0.9);
    const double weight_decay = config.value("weight_decay", 0.0005);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/BCCD_Dataset/BCCD"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/bccd"));

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    LOG_INFO("[BCCD TORCH PIPELINE] Starting on device: {}", device.is_cuda() ? "GPU" : "CPU");
    LOG_INFO("[CONFIG] batch_size={} epochs={} dataset_root={}", batch_size, total_epochs, data_root.string());

    if (device.is_cuda())
    {
        at::globalContext().setBenchmarkCuDNN(true);
    }

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root.string(), train_paths, val_paths, test_paths, BCCD_CLASSES);

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, true, BCCD_CLASSES).map(torch::data::transforms::Stack<>()),
        torch::data::samplers::RandomSampler(train_paths.images.size()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(dataloader_workers));

    auto test_loader = torch::data::make_data_loader(
        VOCYoloDataset(test_paths, false, BCCD_CLASSES).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(dataloader_workers));

    YOLOv1 model(num_classes);
    model->to(device);

    auto get_lr = [&config](int ep) -> float { return scheduled_learning_rate(config, ep); };

    torch::optim::SGD optimizer(model->parameters(),
        torch::optim::SGDOptions(get_lr(1)).momentum(momentum).weight_decay(weight_decay));

    fs::create_directories(results_dir);
    std::ofstream csv_file((results_dir / "metrics_torch.csv").string());
    csv_file << "Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB\n";

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        auto epoch_start_time = std::chrono::steady_clock::now();
        float current_lr = get_lr(epoch);
        for (auto& group : optimizer.param_groups())
        {
            static_cast<torch::optim::SGDOptions&>(group.options()).lr(current_lr);
        }

        model->train();
        float epoch_train_loss = 0.0F;
        int train_batches = 0;

        for (auto& batch : *train_loader)
        {
            auto data = batch.data.to(device, true);
            auto target = batch.target.to(device, true);

            optimizer.zero_grad();
            auto pred = model->forward(data);
            auto loss = compute_yolo_loss(pred, target);

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
            for (auto& batch : *test_loader)
            {
                auto data = batch.data.to(device, true);
                auto target = batch.target.to(device, true);
                auto pred = model->forward(data);
                epoch_test_loss += compute_yolo_loss(pred, target).item().toFloat();
                test_batches++;
            }
        }
        float avg_test_loss = epoch_test_loss / std::max(1, test_batches);

        auto epoch_end_time = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time).count();

        const auto vram = current_vram_mib();
        log_train_epoch("BCCD Torch", epoch, total_epochs, avg_train_loss, avg_test_loss, epoch_duration, vram);
        write_train_test_row(csv_file, epoch, avg_train_loss, avg_test_loss, epoch_duration, vram);
    }

    std::string save_path = (results_dir / "yolov1_bccd_torch_final.pt").string();
    torch::save(model, save_path);
    LOG_INFO("Final model saved: {}", save_path);
    return 0;
}