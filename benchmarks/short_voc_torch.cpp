#include "experiment_config.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "TorchDataset.hpp"
#include "TorchYOLO.hpp"

#include <algorithm>
#include <chrono>
#include <torch/torch.h>

int main()
{
    const nlohmann::json config = load_pipeline_config("voc_torch");
    const auto data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const auto results_dir = resolve_from_source("results/voc_short");
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / "VOC2012").string(), train_paths, val_paths, test_paths);

    auto loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, false).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(16).workers(2));

    YOLOv1 model;
    model->to(device);
    model->train();
    torch::optim::SGD opt(model->parameters(), 1e-5);

    auto csv = open_metrics_csv(results_dir, "metrics_torch.csv", "Epoch;Loss;Time(s);VRAM_MiB");
    constexpr int kEpochs = 3;
    for (int epoch = 1; epoch <= kEpochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        float l_sum = 0.0F;
        int batches = 0;
        for (auto& batch : *loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            opt.zero_grad();
            auto pred = model->forward(data);
            auto loss = compute_yolo_loss(pred, target);
            loss.backward();
            opt.step();
            l_sum += loss.item<float>();
            ++batches;
        }
        const float avg_loss = l_sum / static_cast<float>(std::max(1, batches));
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        const auto vram = current_vram_mib();
        log_train_epoch("Short VOC Torch", epoch, kEpochs, avg_loss, elapsed, vram);
        write_loss_row(csv, epoch, avg_loss, elapsed, vram);
    }
    return 0;
}
