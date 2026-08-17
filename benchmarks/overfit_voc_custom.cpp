#include "experiment_config.hpp"
#include "image_inference.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "YOLO.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/utils.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main()
{
    const nlohmann::json config = load_pipeline_config("overfit_voc_custom");
    apply_pipeline_precision(config);
    const int batch_size = config.value("batch_size", 8);
    const int total_epochs = config.value("epochs", 300);
    const float learning_rate = config.value("learning_rate", 2.0e-5F);
    const int num_classes = config.value("num_classes", 20);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/overfit"));

    LOG_INFO("========================================");
    LOG_INFO("[OVERFIT VOC CUSTOM] Batch: {} | Epochs: {}", batch_size, total_epochs);
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

    CustomDataLoader train_loader(tiny_paths, batch_size, false, VOC_CLASSES);
    YOLO custom_model(num_classes);
    Network trainer(custom_model.get_all_layers(), learning_rate);
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->train();
        layer->learning_rate = learning_rate;
    }

    auto csv_file = open_metrics_csv(results_dir, "metrics_custom.csv", "Epoch;Loss;Time(s);VRAM_MiB");

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        float epoch_loss = 0.0F;
        int batches = 0;
        train_loader.reset();
        while (train_loader.has_next())
        {
            Batch batch = train_loader.get_batch();
            dl::Tensor pred = custom_model.forward(batch.images);
            epoch_loss += YOLOLoss::loss(batch.targets, pred, num_classes).to_host().front();

            dl::Tensor grad_error =
                trainer.clip_loss_gradient(YOLOLoss::loss_derivative(batch.targets, pred, num_classes));
            auto layers = custom_model.get_all_layers();
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {
                grad_error = (*it)->backward(grad_error);
            }
            trainer.clip_parameter_gradients();
            for (auto& layer : layers)
            {
                layer->step();
            }
            ++batches;
        }
        const float avg_loss = epoch_loss / static_cast<float>(std::max(1, batches));
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        const auto vram = current_vram_mib();
        log_train_epoch("Overfit VOC Custom", epoch, total_epochs, avg_loss, elapsed, vram);
        write_loss_row(csv_file, epoch, avg_loss, elapsed, vram);
    }

    const std::string save_path = (results_dir / "yolov1_custom_overfitted.pt").string();
    trainer.save(save_path);
    LOG_INFO("Custom model saved: {}", save_path);

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
        auto raw_det = decode_yolo_tensor(output_data, 0.10F, img.cols, img.rows, num_classes);
        auto final_det = apply_nms(raw_det, 0.45F);
        draw_detections(img, final_det, VOC_CLASSES, cv::Scalar(0, 0, 255));
        cv::imwrite((drawn_dir / fs::path(img_path).filename()).string(), img);
    }
    LOG_INFO("Custom-generated images saved in: {}", drawn_dir.string());
    return 0;
}
