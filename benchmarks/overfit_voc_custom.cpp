#include "experiment_config.hpp"
#include "image_inference.hpp"
#include "prefetch_batch.hpp"
#include "run_metrics.hpp"
#include "yolo_eval.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/mAP.hpp"
#include "DeepLearnLib/utils.hpp"
#include "YOLO.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int kImageSize = 448;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main()
{
    // Read the overfit_voc_custom pipeline JSON so the epoch count, learning rate, and VOC directory come from the overfit experiment.
    const nlohmann::json config = load_pipeline_config("overfit_voc_custom");
    const int batch_size = config.value("batch_size", 8);
    const int total_epochs = config.value("epochs", 300);
    const float learning_rate = config.value("learning_rate", 2.0e-5F);
    const float momentum = config.value("momentum", 0.9F);
    const float weight_decay = config.value("weight_decay", 0.0005F);
    const float gradient_clip = pipeline_gradient_clip(config);
    const int num_classes = config.value("num_classes", 20);
    const float conf_threshold = config.value("conf_threshold", 0.10F);
    const float nms_threshold = config.value("nms_threshold", 0.45F);
    const std::string voc_subset = config.value("voc_subset", "VOC2012");
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/overfit"));

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

    CustomDataLoader train_loader(tiny_paths, batch_size, false, VOC_CLASSES);
    // Build YOLO and the SGD trainer on the GPU so the overfit run goes through the custom stack.
    YOLO custom_model(num_classes);
    Network trainer(custom_model.get_all_layers(), learning_rate, gradient_clip);
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
    }
    apply_sgd_hyperparameters(custom_model.get_all_layers(), learning_rate, momentum, weight_decay);

    log_pipeline_banner({ "Overfit VOC Custom", "custom", batch_size, total_epochs, learning_rate, momentum,
        weight_decay, gradient_clip, num_classes, 0,
        static_cast<int>(custom_model.get_all_layers().size()), data_root.string(), tiny_paths.images.size(),
        tiny_paths.images.size() });

    // Metrics CSV so each epoch appends loss, time, and mAP for comparing runs.
    auto csv_file = open_metrics_csv(results_dir, "metrics_custom.csv", kDetectionCsvHeader);

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        // The custom loader was built with is_train false, so neither pass augments.
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_lr = scheduled_learning_rate(config, epoch);
        apply_sgd_hyperparameters(custom_model.get_all_layers(), current_lr, momentum, weight_decay);
        for (auto& layer : custom_model.get_all_layers())
        {
            layer->train();
        }
        // The training epoch computes the YOLO loss and takes an SGD step so the model memorizes the boxes from this one batch.
        float epoch_loss = 0.0F;
        const int batches = for_each_prefetched_batch(train_loader,
            [&](Batch& batch, int, cudaStream_t stream)
            {
                dl::Tensor pred = custom_model.forward(batch.images, stream);
                epoch_loss += YOLOLoss::loss(batch.targets, pred, num_classes, stream).to_host(stream).front();

                dl::Tensor grad_error = trainer.clip_loss_gradient(
                    YOLOLoss::loss_derivative(batch.targets, pred, num_classes, stream));
                auto layers = custom_model.get_all_layers();
                for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                {
                    grad_error = (*it)->backward(grad_error, stream);
                }
                trainer.clip_parameter_gradients(stream);
                for (auto& layer : layers)
                {
                    layer->step(stream);
                }
            });
        const float avg_loss = epoch_loss / static_cast<float>(std::max(1, batches));
        // Second pass over the same batch in eval mode, so the loss and boxes show how the model memorized these images.
        // Test loss and mAP are a second pass over the same tiny slice after eval, not a held-out split.
        for (auto& layer : custom_model.get_all_layers())
        {
            layer->eval();
        }
        float test_sum = 0.0F;
        std::vector<Detection> predicted_detections;
        std::vector<Detection> ground_truth_detections;
        const int test_batches = for_each_prefetched_batch(train_loader,
            [&](Batch& batch, int, cudaStream_t stream)
            {
                dl::Tensor pred = custom_model.forward(batch.images, stream);
                test_sum += YOLOLoss::loss(batch.targets, pred, num_classes, stream).to_host(stream).front();
                // Detection mAP uses IoU 0.5. Predicted boxes use the JSON confidence and NMS; ground truth uses threshold 0.5 without NMS.
                auto batch_pred = detections_from_tensor(
                    pred, conf_threshold, kImageSize, num_classes, true, nms_threshold, stream);
                auto batch_gt = detections_from_tensor(
                    batch.targets, 0.5F, kImageSize, num_classes, false, nms_threshold, stream);
                predicted_detections.insert(predicted_detections.end(), batch_pred.begin(), batch_pred.end());
                ground_truth_detections.insert(ground_truth_detections.end(), batch_gt.begin(), batch_gt.end());
            });
        const float avg_test = test_sum / static_cast<float>(std::max(1, test_batches));
        // mAP at IoU 0.5 on the decoded boxes, so detection quality is compared at the same threshold in every run.
        const float map50 = mean_average_precision(predicted_detections, ground_truth_detections, 0.5F);
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        log_train_epoch({ "Overfit VOC Custom", epoch, total_epochs, current_lr, avg_loss, avg_test, std::nullopt,
            std::nullopt, map50, batches, elapsed, current_vram_mib() });
        write_detection_row(csv_file, epoch, avg_loss, avg_test, elapsed, current_vram_mib(), map50);
    }

    // Save the weights at the end of the run so the same model can be loaded without repeating training.
    const std::string save_path = (results_dir / "yolov1_custom_overfitted.pt").string();
    trainer.save(save_path);
    log_saved("Overfit VOC Custom", save_path);

    // Draw detections on the images from the overfit batch so the result can be viewed next to the CSV metrics.
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->eval();
    }
    const fs::path drawn_dir = results_dir / "overfit_drawn_custom";
    fs::create_directories(drawn_dir);

    for (const auto& img_path : tiny_paths.images)
    {
        cv::Mat img = cv::imread(img_path);
        if (img.empty())
        {
            continue;
        }
        auto prepared = prepare_yolo_input(img, 448);
        dl::Tensor input = dl::Tensor::from_host({ 1, 3, 448, 448 }, prepared.second.data());
        std::vector<float> output_data = custom_model.forward(input).to_host();
        auto raw_det = decode_yolo_tensor(output_data, conf_threshold, img.cols, img.rows, num_classes);
        auto final_det = apply_nms(raw_det, nms_threshold);
        draw_detections(img, final_det, VOC_CLASSES, cv::Scalar(0, 0, 255));
        cv::imwrite((drawn_dir / fs::path(img_path).filename()).string(), img);
    }
    LOG_INFO("Overfit VOC Custom | done saved={} out={}", tiny_paths.images.size(), drawn_dir.string());
    return 0;
}
