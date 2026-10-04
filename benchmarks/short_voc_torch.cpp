#include "experiment_config.hpp"
#include "run_metrics.hpp"
#include "torch_optim.hpp"
#include "yolo_eval.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/mAP.hpp"
#include "TorchDataset.hpp"
#include "TorchYOLO.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <torch/torch.h>
#include <vector>

int main()
{
    // Read the voc_torch pipeline JSON so the thresholds, learning rate, and VOC directory stay shared with the full experiment.
    const nlohmann::json config = load_pipeline_config("voc_torch");
    const auto data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const auto results_dir = resolve_from_source("results/voc_short");
    // Keep the console log next to the CSV. A new run replaces this file.
    open_results_log(results_dir, "log_torch.txt");
    const std::string voc_subset = config.value("voc_subset", "VOC2012");
    const int batch_size = config.value("batch_size", 16);
    const float learning_rate = config.value("learning_rate", 1.0e-5F);
    const double momentum = config.value("momentum", 0.9);
    const double weight_decay = config.value("weight_decay", 0.0005);
    const float gradient_clip = pipeline_gradient_clip(config);
    const int num_classes = config.value("num_classes", 20);
    const int dataloader_workers = config.value("dataloader_workers", 8);
    const float conf_threshold = config.value("conf_threshold", 0.25F);
    const float nms_threshold = config.value("nms_threshold", 0.5F);
    constexpr int kImageSize = 448;
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Split VOC into image lists so the short run trains on train and computes mAP on test.
    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / voc_subset).string(), train_paths, val_paths, test_paths);

    auto loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, false).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(dataloader_workers));
    auto test_loader = torch::data::make_data_loader(
        VOCYoloDataset(test_paths, false).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(dataloader_workers));

    // Build YOLOv1 in LibTorch so the short run goes through the reference stack.
    YOLOv1 model(num_classes);
    model->to(device);
    model->train();
    auto get_lr = [&config](int ep) -> float
    { return scheduled_learning_rate(config, ep); };
    torch::optim::SGD optimizer = make_sgd(*model, get_lr(1), momentum, weight_decay);

    // The epoch count is fixed in code so this program stays a short check of the VOC loop.
    constexpr int kEpochs = 3;
    log_pipeline_banner({ "Short VOC Torch", "torch", batch_size, kEpochs, learning_rate, static_cast<float>(momentum),
        static_cast<float>(weight_decay), gradient_clip, num_classes,
        dataloader_workers, 0, data_root.string(), train_paths.images.size(), test_paths.images.size() });

    // Metrics CSV so each of the three epochs appends loss, time, and mAP.
    auto csv = open_metrics_csv(results_dir, "metrics_torch.csv", kDetectionCsvHeader);
    for (int epoch = 1; epoch <= kEpochs; ++epoch)
    {
        // The epoch count is the hardcoded kEpochs of 3, and this timer includes the test pass.
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_lr = get_lr(epoch);
        set_sgd_lr(optimizer, current_lr);
        // The training epoch computes the YOLO loss and takes an SGD step so the weights move on the training set.
        float loss_sum = 0.0F;
        int batches = 0;
        for (auto& batch : *loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            optimizer.zero_grad();
            auto pred = model->forward(data);
            auto loss = compute_yolo_loss(pred, target);
            loss.backward();
            clip_torch_grad_value(*model, gradient_clip);
            optimizer.step();
            loss_sum += loss.item<float>();
            ++batches;
        }
        const float avg_loss = loss_sum / static_cast<float>(std::max(1, batches));

        // Test pass in eval mode, so the short run collects loss and boxes for mAP.
        model->eval();
        float test_sum = 0.0F;
        int test_batches = 0;
        std::vector<Detection> predicted_detections;
        std::vector<Detection> ground_truth_detections;
        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *test_loader)
            {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                auto pred = model->forward(data);
                test_sum += compute_yolo_loss(pred, target).item<float>();
                ++test_batches;
                const auto pred_host = tensor_to_host_f32(pred);
                const auto target_host = tensor_to_host_f32(target);
                const int batch_n = static_cast<int>(pred.size(0));
                const int elems = static_cast<int>(pred.numel() / std::max<int64_t>(pred.size(0), 1));
                // Detection mAP uses IoU 0.5. Predicted boxes use the JSON confidence and NMS; ground truth uses threshold 0.5 without NMS.
                auto batch_pred = detections_from_flat(
                    pred_host, batch_n, elems, conf_threshold, kImageSize, num_classes, true, nms_threshold);
                auto batch_gt = detections_from_flat(
                    target_host, batch_n, elems, 0.5F, kImageSize, num_classes, false, nms_threshold);
                predicted_detections.insert(predicted_detections.end(), batch_pred.begin(), batch_pred.end());
                ground_truth_detections.insert(ground_truth_detections.end(), batch_gt.begin(), batch_gt.end());
            }
        }
        model->train();
        const float avg_test = test_sum / static_cast<float>(std::max(1, test_batches));
        // mAP at IoU 0.5 on the decoded boxes, so detection quality is compared at the same threshold in every run.
        const float map50 = mean_average_precision(predicted_detections, ground_truth_detections, 0.5F);
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        log_train_epoch({ "Short VOC Torch", epoch, kEpochs, current_lr, avg_loss, avg_test, std::nullopt,
            std::nullopt, map50, batches, elapsed, current_vram_mib() });
        write_detection_row(csv, epoch, avg_loss, avg_test, elapsed, current_vram_mib(), map50);
    }
    return 0;
}
