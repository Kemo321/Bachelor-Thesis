#include "experiment_config.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/mAP.hpp"
#include "DeepLearnLib/utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

constexpr int kImageSize = 448;

auto slice_sample(const std::vector<float>& host, int sample_index, int elements_per_sample) -> std::vector<float>
{
    const auto offset = static_cast<std::size_t>(sample_index) * static_cast<std::size_t>(elements_per_sample);
    return { host.begin() + static_cast<std::ptrdiff_t>(offset),
        host.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<std::size_t>(elements_per_sample)) };
}

auto detections_from_batch(const dl::Tensor& tensor, float conf_threshold, int num_classes, bool apply_suppression,
    float nms_threshold) -> std::vector<Detection>
{
    const std::vector<float> host = tensor.to_host();
    const int batch = tensor.get_shape()[0];
    const int elements_per_sample = static_cast<int>(tensor.get_size()) / batch;
    std::vector<Detection> all;
    for (int sample = 0; sample < batch; ++sample)
    {
        std::vector<float> sample_buffer = slice_sample(host, sample, elements_per_sample);
        std::vector<Detection> decoded = decode_yolo_tensor(sample_buffer, conf_threshold, kImageSize, kImageSize,
            num_classes);
        if (apply_suppression)
        {
            decoded = apply_nms(decoded, nms_threshold);
        }
        all.insert(all.end(), decoded.begin(), decoded.end());
    }
    return all;
}

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const nlohmann::json config = load_pipeline_config("voc_custom");
    const int batch_size = config.value("batch_size", 16);
    const int total_epochs = config.value("epochs", 150);
    const float learning_rate = config.value("learning_rate", 1.0e-4F);
    const int num_classes = config.value("num_classes", 20);
    const float conf_threshold = config.value("conf_threshold", 0.25F);
    const float nms_threshold = config.value("nms_threshold", 0.5F);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/voc"));

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    LOG_INFO("[VOC CUSTOM PIPELINE] Starting on device: {}", gpu_count > 0 ? "GPU" : "CPU");
    LOG_INFO("[CONFIG] batch_size={} epochs={} learning_rate={} dataset_root={}", batch_size, total_epochs,
        learning_rate, data_root.string());

    DataPaths train_paths, val_paths, test_paths;
    split_dataset((data_root / "VOC2012").string(), train_paths, val_paths, test_paths, VOC_CLASSES);

    CustomDataLoader train_loader(train_paths, batch_size, true, VOC_CLASSES);
    CustomDataLoader test_loader(test_paths, batch_size, false, VOC_CLASSES);

    YOLO custom_model(num_classes);
    Network trainer(custom_model.get_all_layers(), learning_rate);

    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
    }

    auto get_lr = [learning_rate](int ep) -> float
    {
        if (ep <= 5)
            return learning_rate * 0.1F;
        if (ep <= 80)
            return learning_rate;
        if (ep <= 120)
            return learning_rate * 0.1F;
        return learning_rate * 0.01F;
    };

    fs::create_directories(results_dir);
    std::ofstream csv_file((results_dir / "metrics_custom.csv").string());
    csv_file << "Epoch;TrainLoss;TestLoss;mAP@0.5;Time(s)\n";

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        auto epoch_start_time = std::chrono::steady_clock::now();
        float current_lr = get_lr(epoch);

        for (auto& layer : custom_model.get_all_layers())
        {
            layer->learning_rate = current_lr;
            layer->train();
        }

        float epoch_train_loss = 0.0F;
        int train_batches = 0;

        train_loader.reset();
        while (train_loader.has_next())
        {
            Batch batch = train_loader.get_batch();

            dl::Tensor pred = custom_model.forward(batch.images);
            const float batch_loss = YOLOLoss::loss(batch.targets, pred, num_classes).to_host().front();

            dl::Tensor grad_error = YOLOLoss::loss_derivative(batch.targets, pred, num_classes);
            grad_error = grad_error.clamp(-10.0F, 10.0F);

            auto layers = custom_model.get_all_layers();
            for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
            {
                grad_error = (*iterator)->backward(grad_error);
            }
            for (auto& layer : layers)
            {
                layer->step();
            }

            epoch_train_loss += batch_loss;
            train_batches++;
        }
        float avg_train_loss = epoch_train_loss / static_cast<float>(std::max(1, train_batches));

        for (auto& layer : custom_model.get_all_layers())
        {
            layer->eval();
        }

        float epoch_test_loss = 0.0F;
        int test_batches = 0;
        std::vector<Detection> predicted_detections;
        std::vector<Detection> ground_truth_detections;

        test_loader.reset();
        while (test_loader.has_next())
        {
            Batch batch = test_loader.get_batch();
            dl::Tensor pred = custom_model.forward(batch.images);
            epoch_test_loss += YOLOLoss::loss(batch.targets, pred, num_classes).to_host().front();
            test_batches++;

            auto batch_pred = detections_from_batch(pred, conf_threshold, num_classes, true, nms_threshold);
            auto batch_gt = detections_from_batch(batch.targets, 0.5F, num_classes, false, nms_threshold);
            predicted_detections.insert(predicted_detections.end(), batch_pred.begin(), batch_pred.end());
            ground_truth_detections.insert(ground_truth_detections.end(), batch_gt.begin(), batch_gt.end());
        }
        float avg_test_loss = epoch_test_loss / static_cast<float>(std::max(1, test_batches));
        const float map50 = mean_average_precision(predicted_detections, ground_truth_detections, 0.5F);

        auto epoch_end_time = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time).count();

        LOG_INFO("VOC Custom | Epoch [{}/{}] | Train Loss: {:.4f} | Test Loss: {:.4f} | mAP@0.5: {} | Time: {}s", epoch,
            total_epochs, avg_train_loss, avg_test_loss, map50, epoch_duration);

        csv_file << epoch << ";" << avg_train_loss << ";" << avg_test_loss << ";" << map50 << ";" << epoch_duration
                 << "\n";
        csv_file.flush();
    }

    std::string save_path = (results_dir / "yolov1_voc_custom_final.pt").string();
    trainer.save(save_path);
    LOG_INFO("Final model saved: {}", save_path);
    return 0;
}
