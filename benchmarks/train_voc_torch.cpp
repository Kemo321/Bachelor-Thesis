#include "experiment_config.hpp"
#include "run_metrics.hpp"
#include "torch_optim.hpp"
#include "yolo_eval.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/mAP.hpp"
#include "TorchDataset.hpp"
#include "TorchYOLO.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

constexpr int kImageSize = 448;

int main()
{
    std::srand(std::time(nullptr));

    const nlohmann::json config = load_pipeline_config("voc_torch");
    apply_pipeline_precision(config);
    const int batch_size = config.value("batch_size", 16);
    const int total_epochs = config.value("epochs", 150);
    const int num_classes = config.value("num_classes", 20);
    const int dataloader_workers = config.value("dataloader_workers", 8);
    const float learning_rate = config.value("learning_rate", 1.0e-4F);
    const double momentum = config.value("momentum", 0.9);
    const double weight_decay = config.value("weight_decay", 0.0005);
    const float gradient_clip = pipeline_gradient_clip(config);
    const float conf_threshold = config.value("conf_threshold", 0.25F);
    const float nms_threshold = config.value("nms_threshold", 0.5F);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/voc"));
    const std::string voc_subset = config.value("voc_subset", "VOC2012");

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    if (device.is_cuda())
    {
        at::globalContext().setBenchmarkCuDNN(true);
    }

    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / voc_subset).string(), train_paths, val_paths, test_paths, VOC_CLASSES);

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, true, VOC_CLASSES).map(torch::data::transforms::Stack<>()),
        torch::data::samplers::RandomSampler(train_paths.images.size()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(dataloader_workers));

    auto test_loader = torch::data::make_data_loader(
        VOCYoloDataset(test_paths, false, VOC_CLASSES).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(dataloader_workers));

    YOLOv1 model(num_classes);
    model->to(device);

    auto get_lr = [&config](int ep) -> float
    { return scheduled_learning_rate(config, ep); };

    torch::optim::SGD optimizer = make_sgd(*model, get_lr(1), momentum, weight_decay);

    log_pipeline_banner({ "VOC Torch", "torch", batch_size, total_epochs, learning_rate, static_cast<float>(momentum),
        static_cast<float>(weight_decay), gradient_clip, pipeline_precision_name(config), num_classes,
        dataloader_workers, 0, data_root.string(), train_paths.images.size(), test_paths.images.size() });

    fs::create_directories(results_dir);
    std::ofstream csv_file((results_dir / "metrics_torch.csv").string());
    csv_file << "Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;mAP@0.5\n";

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        auto epoch_start_time = std::chrono::steady_clock::now();
        float current_lr = get_lr(epoch);
        set_sgd_lr(optimizer, current_lr);

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
            clip_torch_grad_value(*model, gradient_clip);
            optimizer.step();

            epoch_train_loss += loss.item().toFloat();
            train_batches++;
        }
        float avg_train_loss = epoch_train_loss / std::max(1, train_batches);

        model->eval();
        float epoch_test_loss = 0.0F;
        int test_batches = 0;
        std::vector<Detection> predicted_detections;
        std::vector<Detection> ground_truth_detections;

        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *test_loader)
            {
                auto data = batch.data.to(device, true);
                auto target = batch.target.to(device, true);
                auto pred = model->forward(data);
                epoch_test_loss += compute_yolo_loss(pred, target).item().toFloat();
                test_batches++;

                const auto pred_host = tensor_to_host_f32(pred);
                const auto target_host = tensor_to_host_f32(target);
                const int batch_n = static_cast<int>(pred.size(0));
                const int elems = static_cast<int>(pred.numel() / pred.size(0));
                auto batch_pred = detections_from_flat(
                    pred_host, batch_n, elems, conf_threshold, kImageSize, num_classes, true, nms_threshold);
                auto batch_gt = detections_from_flat(
                    target_host, batch_n, elems, 0.5F, kImageSize, num_classes, false, nms_threshold);
                predicted_detections.insert(predicted_detections.end(), batch_pred.begin(), batch_pred.end());
                ground_truth_detections.insert(ground_truth_detections.end(), batch_gt.begin(), batch_gt.end());
            }
        }
        float avg_test_loss = epoch_test_loss / std::max(1, test_batches);
        const float map50 = mean_average_precision(predicted_detections, ground_truth_detections, 0.5F);

        auto epoch_end_time = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time).count();

        log_train_epoch({ "VOC Torch", epoch, total_epochs, current_lr, avg_train_loss, avg_test_loss, std::nullopt,
            std::nullopt, map50, train_batches, epoch_duration, current_vram_mib() });
        csv_file << epoch << ";" << avg_train_loss << ";" << avg_test_loss << ";" << epoch_duration << ";"
                 << current_vram_mib() << ";" << map50 << "\n";
        csv_file.flush();
    }

    std::string save_path = (results_dir / "yolov1_voc_torch_final.pt").string();
    torch::save(model, save_path);
    log_saved("VOC Torch", save_path);
    return 0;
}
