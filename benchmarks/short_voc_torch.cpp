#include "experiment_config.hpp"
#include "run_metrics.hpp"
#include "torch_optim.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "TorchDataset.hpp"
#include "TorchYOLO.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <torch/torch.h>

int main()
{
    const nlohmann::json config = load_pipeline_config("voc_torch");
    apply_pipeline_precision(config);
    const auto data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const auto results_dir = resolve_from_source("results/voc_short");
    const std::string voc_subset = config.value("voc_subset", "VOC2012");
    const int batch_size = config.value("batch_size", 16);
    const float learning_rate = config.value("learning_rate", 1.0e-5F);
    const double momentum = config.value("momentum", 0.9);
    const double weight_decay = config.value("weight_decay", 0.0005);
    const float gradient_clip = pipeline_gradient_clip(config);
    const int num_classes = config.value("num_classes", 20);
    const int dataloader_workers = config.value("dataloader_workers", 8);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / voc_subset).string(), train_paths, val_paths, test_paths);

    auto loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, false).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(dataloader_workers));

    YOLOv1 model(num_classes);
    model->to(device);
    model->train();
    auto get_lr = [&config](int ep) -> float
    { return scheduled_learning_rate(config, ep); };
    torch::optim::SGD opt = make_sgd(*model, get_lr(1), momentum, weight_decay);

    constexpr int kEpochs = 3;
    log_pipeline_banner({ "Short VOC Torch", "torch", batch_size, kEpochs, learning_rate, static_cast<float>(momentum),
        static_cast<float>(weight_decay), gradient_clip, pipeline_precision_name(config), num_classes,
        dataloader_workers, 0, data_root.string(), train_paths.images.size(), 0 });

    auto csv = open_metrics_csv(results_dir, "metrics_torch.csv", "Epoch;Loss;Time(s);VRAM_MiB");
    for (int epoch = 1; epoch <= kEpochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_lr = get_lr(epoch);
        set_sgd_lr(opt, current_lr);
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
            clip_torch_grad_value(*model, gradient_clip);
            opt.step();
            l_sum += loss.item<float>();
            ++batches;
        }
        const float avg_loss = l_sum / static_cast<float>(std::max(1, batches));
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        log_train_epoch({ "Short VOC Torch", epoch, kEpochs, current_lr, avg_loss, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, batches, elapsed, current_vram_mib() });
        write_loss_row(csv, epoch, avg_loss, elapsed, current_vram_mib());
    }
    return 0;
}
