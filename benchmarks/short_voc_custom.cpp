#include "experiment_config.hpp"
#include "prefetch_batch.hpp"
#include "run_metrics.hpp"
#include "yolo_eval.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/mAP.hpp"
#include "YOLO.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <vector>

int main()
{
    // Read the voc_custom pipeline JSON so the thresholds, learning rate, and VOC directory stay shared with the full experiment.
    const nlohmann::json config = load_pipeline_config("voc_custom");
    const auto data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const auto results_dir = resolve_from_source("results/voc_short");
    // Keep the console log next to the CSV. A new run replaces this file.
    open_results_log(results_dir, "log_custom.txt");
    const std::string voc_subset = config.value("voc_subset", "VOC2012");
    const int batch_size = config.value("batch_size", 16);
    const float learning_rate = config.value("learning_rate", 1.0e-5F);
    const float momentum = config.value("momentum", 0.9F);
    const float weight_decay = config.value("weight_decay", 0.0005F);
    const float gradient_clip = pipeline_gradient_clip(config);
    const int num_classes = config.value("num_classes", 20);
    const float conf_threshold = config.value("conf_threshold", 0.25F);
    const float nms_threshold = config.value("nms_threshold", 0.5F);
    constexpr int kImageSize = 448;

    // Split VOC into image lists so the short run trains on train and computes mAP on test.
    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / voc_subset).string(), train_paths, val_paths, test_paths);

    CustomDataLoader loader(train_paths, batch_size, false);
    CustomDataLoader test_loader(test_paths, batch_size, false);
    // Build YOLO and the SGD trainer on the GPU so the short run goes through the custom stack.
    YOLO custom_model(num_classes);
    Network trainer(custom_model.get_all_layers(), learning_rate, gradient_clip);
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
    }
    apply_sgd_hyperparameters(custom_model.get_all_layers(), learning_rate, momentum, weight_decay);

    // The epoch count is fixed in code so this program stays a short check of the VOC loop.
    constexpr int kEpochs = 3;
    log_pipeline_banner({ "Short VOC Custom", "custom", batch_size, kEpochs, learning_rate, momentum, weight_decay,
        gradient_clip, num_classes, 0,
        static_cast<int>(custom_model.get_all_layers().size()), data_root.string(), train_paths.images.size(),
        test_paths.images.size() });

    // Metrics CSV so each of the three epochs appends loss, time, and mAP.
    auto csv = open_metrics_csv(results_dir, "metrics_custom.csv", kDetectionCsvHeader);
    for (int epoch = 1; epoch <= kEpochs; ++epoch)
    {
        // The epoch count is the hardcoded kEpochs of 3, and this timer includes the test pass.
        // Both custom loaders were built with is_train false, so neither pass augments.
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_lr = scheduled_learning_rate(config, epoch);
        apply_sgd_hyperparameters(custom_model.get_all_layers(), current_lr, momentum, weight_decay);
        for (auto& layer : custom_model.get_all_layers())
        {
            layer->train();
        }
        // The training epoch computes the YOLO loss and takes an SGD step so the weights move on the training set.
        float loss_sum = 0.0F;
        const int batches = for_each_prefetched_batch(loader,
            [&](Batch& batch, int, cudaStream_t stream)
            {
                dl::Tensor pred = custom_model.forward(batch.images, stream);
                loss_sum += YOLOLoss::loss(batch.targets, pred, num_classes, stream).to_host(stream).front();

                dl::Tensor grad = trainer.clip_loss_gradient(
                    YOLOLoss::loss_derivative(batch.targets, pred, num_classes, stream));
                auto layers = custom_model.get_all_layers();
                for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                {
                    grad = (*it)->backward(grad, stream);
                }
                trainer.clip_parameter_gradients(stream);
                for (auto& layer : layers)
                {
                    layer->step(stream);
                }
            });
        const float avg_loss = loss_sum / static_cast<float>(std::max(1, batches));

        // Test pass in eval mode, so the short run collects loss and boxes for mAP.
        for (auto& layer : custom_model.get_all_layers())
        {
            layer->eval();
        }
        float test_sum = 0.0F;
        std::vector<Detection> predicted_detections;
        std::vector<Detection> ground_truth_detections;
        const int test_batches = for_each_prefetched_batch(test_loader,
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
        log_train_epoch({ "Short VOC Custom", epoch, kEpochs, current_lr, avg_loss, avg_test, std::nullopt,
            std::nullopt, map50, batches, elapsed, current_vram_mib() });
        write_detection_row(csv, epoch, avg_loss, avg_test, elapsed, current_vram_mib(), map50);
    }
    return 0;
}
