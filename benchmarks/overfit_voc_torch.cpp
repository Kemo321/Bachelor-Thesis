#include "experiment_config.hpp"
#include "image_inference.hpp"
#include "run_metrics.hpp"
#include "torch_optim.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/utils.hpp"
#include "TorchDataset.hpp"
#include "TorchYOLO.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main()
{
    const nlohmann::json config = load_pipeline_config("overfit_voc_torch");
    apply_pipeline_precision(config);
    const int batch_size = config.value("batch_size", 8);
    const int total_epochs = config.value("epochs", 300);
    const float learning_rate = config.value("learning_rate", 2.0e-5F);
    const double momentum = config.value("momentum", 0.9);
    const double weight_decay = config.value("weight_decay", 0.0005);
    const float gradient_clip = pipeline_gradient_clip(config);
    const int num_classes = config.value("num_classes", 20);
    const int dataloader_workers = config.value("dataloader_workers", 0);
    const float conf_threshold = config.value("conf_threshold", 0.10F);
    const float nms_threshold = config.value("nms_threshold", 0.45F);
    const std::string voc_subset = config.value("voc_subset", "VOC2012");
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/overfit"));

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / voc_subset).string(), train_paths, val_paths, test_paths, VOC_CLASSES);
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
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(dataloader_workers));

    YOLOv1 model(num_classes);
    model->to(device);
    model->train();
    auto get_lr = [&config](int ep) -> float
    { return scheduled_learning_rate(config, ep); };
    torch::optim::SGD optimizer = make_sgd(*model, get_lr(1), momentum, weight_decay);

    log_pipeline_banner({ "Overfit VOC Torch", "torch", batch_size, total_epochs, learning_rate,
        static_cast<float>(momentum), static_cast<float>(weight_decay), gradient_clip,
        pipeline_precision_name(config), num_classes, dataloader_workers, 0, data_root.string(),
        tiny_paths.images.size(), 0 });

    auto csv_file = open_metrics_csv(results_dir, "metrics_torch.csv", "Epoch;Loss;Time(s);VRAM_MiB");

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_lr = get_lr(epoch);
        set_sgd_lr(optimizer, current_lr);
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
            clip_torch_grad_value(*model, gradient_clip);
            optimizer.step();
            epoch_loss += loss.item().toFloat();
            ++batches;
        }
        const float avg_loss = epoch_loss / static_cast<float>(std::max(1, batches));
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        log_train_epoch({ "Overfit VOC Torch", epoch, total_epochs, current_lr, avg_loss, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, batches, elapsed, current_vram_mib() });
        write_loss_row(csv_file, epoch, avg_loss, elapsed, current_vram_mib());
    }

    const std::string save_path = (results_dir / "yolov1_torch_overfitted.pt").string();
    torch::save(model, save_path);
    log_saved("Overfit VOC Torch", save_path);

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
        auto raw_det = decode_yolo_tensor(tensor_to_host_f32(output), conf_threshold, img.cols, img.rows, num_classes);
        auto final_det = apply_nms(raw_det, nms_threshold);
        draw_detections(img, final_det, VOC_CLASSES, cv::Scalar(0, 255, 0));
        cv::imwrite((drawn_dir / fs::path(img_path).filename()).string(), img);
    }
    LOG_INFO("Overfit VOC Torch | done saved={} out={}", tiny_paths.images.size(), drawn_dir.string());
    return 0;
}
