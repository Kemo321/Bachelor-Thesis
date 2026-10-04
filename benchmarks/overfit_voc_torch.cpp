#include "experiment_config.hpp"
#include "image_inference.hpp"
#include "run_metrics.hpp"
#include "torch_optim.hpp"
#include "yolo_eval.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/mAP.hpp"
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

constexpr int kImageSize = 448;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main()
{
    // Read the overfit_voc_torch pipeline JSON so the epoch count, learning rate, and VOC directory come from the overfit experiment.
    const nlohmann::json config = load_pipeline_config("overfit_voc_torch");
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

    // Split VOC, then cut it down to at most one batch, so the model can overfit a few images.
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

    // Build YOLOv1 in LibTorch so the overfit run goes through the reference stack.
    YOLOv1 model(num_classes);
    model->to(device);
    auto get_lr = [&config](int ep) -> float
    { return scheduled_learning_rate(config, ep); };
    torch::optim::SGD optimizer = make_sgd(*model, get_lr(1), momentum, weight_decay);

    log_pipeline_banner({ "Overfit VOC Torch", "torch", batch_size, total_epochs, learning_rate,
        static_cast<float>(momentum), static_cast<float>(weight_decay), gradient_clip,
        num_classes, dataloader_workers, 0, data_root.string(),
        tiny_paths.images.size(), tiny_paths.images.size() });

    // Metrics CSV so each epoch appends loss, time, and mAP for comparing runs.
    auto csv_file = open_metrics_csv(results_dir, "metrics_torch.csv", kDetectionCsvHeader);

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_lr = get_lr(epoch);
        set_sgd_lr(optimizer, current_lr);
        // The training epoch in train mode computes the YOLO loss and takes an SGD step so the model memorizes the boxes from this one batch.
        model->train();
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
        // Second pass over the same batch in eval mode, so the loss and boxes show how the model memorized these images.
        // Test loss and mAP are a second pass over the same tiny slice after eval, not a held-out split.
        model->eval();
        float test_sum = 0.0F;
        int test_batches = 0;
        std::vector<Detection> predicted_detections;
        std::vector<Detection> ground_truth_detections;
        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *train_loader)
            {
                auto data = batch.data.to(device, true);
                auto target = batch.target.to(device, true);
                auto pred = model->forward(data);
                test_sum += compute_yolo_loss(pred, target).item().toFloat();
                ++test_batches;
                const auto pred_host = tensor_to_host_f32(pred);
                const auto target_host = tensor_to_host_f32(target);
                const int batch_n = static_cast<int>(pred.size(0));
                const int elems = static_cast<int>(pred.numel() / pred.size(0));
                // Detection mAP uses IoU 0.5. Predicted boxes use the JSON confidence and NMS; ground truth uses threshold 0.5 without NMS.
                auto batch_pred = detections_from_flat(
                    pred_host, batch_n, elems, conf_threshold, kImageSize, num_classes, true, nms_threshold);
                auto batch_gt = detections_from_flat(
                    target_host, batch_n, elems, 0.5F, kImageSize, num_classes, false, nms_threshold);
                predicted_detections.insert(predicted_detections.end(), batch_pred.begin(), batch_pred.end());
                ground_truth_detections.insert(ground_truth_detections.end(), batch_gt.begin(), batch_gt.end());
            }
        }
        const float avg_test = test_sum / static_cast<float>(std::max(1, test_batches));
        // mAP at IoU 0.5 on the decoded boxes, so detection quality is compared at the same threshold in every run.
        const float map50 = mean_average_precision(predicted_detections, ground_truth_detections, 0.5F);
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        log_train_epoch({ "Overfit VOC Torch", epoch, total_epochs, current_lr, avg_loss, avg_test, std::nullopt,
            std::nullopt, map50, batches, elapsed, current_vram_mib() });
        write_detection_row(csv_file, epoch, avg_loss, avg_test, elapsed, current_vram_mib(), map50);
    }

    // Save the weights at the end of the run so the same model can be loaded without repeating training.
    const std::string save_path = (results_dir / "yolov1_torch_overfitted.pt").string();
    torch::save(model, save_path);
    log_saved("Overfit VOC Torch", save_path);

    // Draw detections on the images from the overfit batch so the result can be viewed next to the CSV metrics.
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
