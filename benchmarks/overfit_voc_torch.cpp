#include "experiment_config.hpp"
#include "image_inference.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/utils.hpp"
#include "TorchDataset.hpp"
#include "TorchYOLO.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

auto torch_to_host(const torch::Tensor& tensor) -> std::vector<float>
{
    const auto cpu = tensor.contiguous().to(torch::kCPU).to(torch::kFloat32);
    std::vector<float> host(static_cast<std::size_t>(cpu.numel()));
    std::memcpy(host.data(), cpu.data_ptr<float>(), host.size() * sizeof(float));
    return host;
}

int main()
{
    const nlohmann::json config = load_pipeline_config("overfit_voc_torch");
    const int batch_size = config.value("batch_size", 8);
    const int total_epochs = config.value("epochs", 300);
    const float learning_rate = config.value("learning_rate", 2.0e-5F);
    const int num_classes = config.value("num_classes", 20);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/overfit"));

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    LOG_INFO("========================================");
    LOG_INFO("[OVERFIT VOC TORCH] Device: {} | Batch: {} | Epochs: {}", device.is_cuda() ? "GPU" : "CPU", batch_size,
        total_epochs);
    LOG_INFO("========================================");

    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / "VOC2012").string(), train_paths, val_paths, test_paths, VOC_CLASSES);
    if (train_paths.images.empty())
    {
        LOG_ERROR("No data in the data folder!");
        return 1;
    }

    DataPaths tiny_paths;
    for (int i = 0; i < batch_size && i < static_cast<int>(train_paths.images.size()); ++i)
    {
        tiny_paths.images.push_back(train_paths.images[i]);
        tiny_paths.labels.push_back(train_paths.labels[i]);
    }

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(tiny_paths, false, VOC_CLASSES).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(0));

    YOLOv1 model(num_classes);
    model->to(device);
    model->train();
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    auto csv_file = open_metrics_csv(results_dir, "metrics_torch.csv", "Epoch;Loss;Time(s);VRAM_MiB");

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        float epoch_loss = 0.0F;
        int batches = 0;
        for (auto& batch : *train_loader)
        {
            auto data = batch.data.to(device, true);
            auto target = batch.target.to(device, true);
            optimizer.zero_grad();
            auto pred = model->forward(data);
            auto loss = compute_yolo_loss(pred, target);
            loss.backward();
            optimizer.step();
            epoch_loss += loss.item().toFloat();
            ++batches;
        }
        const float avg_loss = epoch_loss / static_cast<float>(std::max(1, batches));
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        const auto vram = current_vram_mib();
        log_train_epoch("Overfit VOC Torch", epoch, total_epochs, avg_loss, elapsed, vram);
        write_loss_row(csv_file, epoch, avg_loss, elapsed, vram);
    }

    const std::string save_path = (results_dir / "yolov1_torch_overfitted.pt").string();
    torch::save(model, save_path);
    LOG_INFO("Torch model saved: {}", save_path);

    model->eval();
    const fs::path drawn_dir = results_dir / "overfit_drawn_torch";
    fs::create_directories(drawn_dir);
    for (const auto& img_path : tiny_paths.images)
    {
        cv::Mat img = cv::imread(img_path);
        if (img.empty())
        {
            continue;
        }
        auto prepared = prepare_yolo_input(img, 448);
        auto input = torch::from_blob(prepared.second.data(), { 1, 3, 448, 448 }, torch::kFloat32).clone().to(device);
        torch::Tensor output;
        {
            torch::NoGradGuard no_grad;
            output = model->forward(input);
        }
        auto raw_det = decode_yolo_tensor(torch_to_host(output), 0.10F, img.cols, img.rows, num_classes);
        auto final_det = apply_nms(raw_det, 0.45F);
        draw_detections(img, final_det, VOC_CLASSES, cv::Scalar(0, 255, 0));
        cv::imwrite((drawn_dir / fs::path(img_path).filename()).string(), img);
    }
    LOG_INFO("Torch-generated images saved in: {}", drawn_dir.string());
    return 0;
}
