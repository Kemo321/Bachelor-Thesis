#include "experiment_config.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/DarknetDetectionLoss.hpp"
#include "DeepLearnLib/DarknetWeights.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/mAP.hpp"
#include "DeepLearnLib/utils.hpp"
#include "YOLODarknet.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
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

auto predicted_from_batch(const dl::Tensor& tensor, float conf_threshold, int num_classes, float nms_threshold)
    -> std::vector<Detection>
{
    const std::vector<float> host = tensor.to_host();
    const int batch = tensor.get_shape()[0];
    const int elements_per_sample = static_cast<int>(tensor.get_size()) / batch;
    std::vector<Detection> all;
    for (int sample = 0; sample < batch; ++sample)
    {
        std::vector<float> sample_buffer = slice_sample(host, sample, elements_per_sample);
        std::vector<Detection> decoded = decode_darknet_detection(sample_buffer, conf_threshold, kImageSize, kImageSize,
            num_classes, YOLODarknet::kGridSize, YOLODarknet::kBoxesPerCell, true);
        decoded = apply_nms(decoded, nms_threshold);
        all.insert(all.end(), decoded.begin(), decoded.end());
    }
    return all;
}

auto truth_from_batch(const dl::Tensor& tensor, int num_classes) -> std::vector<Detection>
{
    const std::vector<float> host = tensor.to_host();
    const int batch = tensor.get_shape()[0];
    const int elements_per_sample = static_cast<int>(tensor.get_size()) / batch;
    std::vector<Detection> all;
    for (int sample = 0; sample < batch; ++sample)
    {
        std::vector<float> sample_buffer = slice_sample(host, sample, elements_per_sample);
        auto decoded = detections_from_darknet_truth(sample_buffer, kImageSize, kImageSize, num_classes,
            YOLODarknet::kGridSize);
        all.insert(all.end(), decoded.begin(), decoded.end());
    }
    return all;
}

auto first_unfrozen_index(const std::vector<std::shared_ptr<Layer>>& layers) -> int
{
    for (int index = 0; index < static_cast<int>(layers.size()); ++index)
    {
        if (!layers[static_cast<std::size_t>(index)]->frozen())
        {
            return index;
        }
    }
    return static_cast<int>(layers.size());
}

int main()
{
    try
    {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));

        const nlohmann::json config = load_pipeline_config("voc_darknet_custom");
        apply_pipeline_precision(config);
        const int batch_size = config.value("batch_size", 8);
        const int total_epochs = config.value("epochs", 40);
        const float learning_rate = config.value("learning_rate", 5.0e-4F);
        const float momentum = config.value("momentum", 0.9F);
        const float gradient_clip = pipeline_gradient_clip(config);
        const int num_classes = config.value("num_classes", 20);
        const float conf_threshold = config.value("conf_threshold", 0.2F);
        const float nms_threshold = config.value("nms_threshold", 0.4F);
        const bool freeze_backbone = config.value("freeze_backbone", true);
        const bool require_weights = config.value("require_weights", true);
        const int cutoff_convs = config.value("darknet_cutoff_convs", 20);
        const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/VOCdevkit"));
        const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/voc_darknet"));
        const fs::path weights_path = resolve_from_source(
            config.value("darknet_weights", "data/darknet/extraction.conv.weights"));

        int gpu_count = 0;
        cudaGetDeviceCount(&gpu_count);
        LOG_INFO("[VOC DARKNET PIPELINE] Starting on device: {}", gpu_count > 0 ? "GPU" : "CPU");
        LOG_INFO("[CONFIG] batch_size={} epochs={} lr={} momentum={} freeze_backbone={} weights={}", batch_size,
            total_epochs, learning_rate, momentum, freeze_backbone, weights_path.string());

        DataPaths train_paths, val_paths, test_paths;
        split_dataset((data_root / "VOC2012").string(), train_paths, val_paths, test_paths, VOC_CLASSES);

        CustomDataLoader train_loader(train_paths, batch_size, true, VOC_CLASSES,
            DetectionLabelLayout::DarknetYolov1);
        CustomDataLoader test_loader(test_paths, batch_size, false, VOC_CLASSES,
            DetectionLabelLayout::DarknetYolov1);

        YOLODarknet custom_model(num_classes);
        Network trainer(custom_model.get_all_layers(), learning_rate, gradient_clip);

        for (auto& layer : custom_model.get_all_layers())
        {
            layer->to(dl::Device::GPU);
            layer->momentum = momentum;
        }

        if (fs::exists(weights_path))
        {
            DarknetLoadOptions options;
            options.cutoff_convs = cutoff_convs;
            options.load_local = config.value("load_local", cutoff_convs <= 0);
            options.load_connected = config.value("load_connected", cutoff_convs <= 0);
            const auto report = load_darknet_weights(custom_model.get_all_layers(), weights_path.string(), options);
            if (report.bytes_remaining > 0 && cutoff_convs <= 0)
            {
                LOG_INFO("Darknet weights have {} leftover bytes (expected 0 for a full yolov1.weights load)",
                    report.bytes_remaining);
            }
        }
        else if (require_weights)
        {
            throw std::runtime_error("Darknet weights not found: " + weights_path.string()
                + " (run scripts/setup_datasets.py --only darknet)");
        }
        else
        {
            LOG_INFO("Darknet weights missing at {}; training from random init", weights_path.string());
        }

        if (freeze_backbone)
        {
            custom_model.freeze_extraction_backbone();
            LOG_INFO("Froze ImageNet extraction backbone (first 20 convs); training detection head");
        }

        LOG_INFO("YOLODarknet on GPU ({} layers). Train images={} test={} pred_size={}",
            custom_model.get_all_layers().size(), train_loader.size(), test_loader.size(),
            YOLODarknet::prediction_size(num_classes));
        LOG_FLUSH();

        DarknetDetectionLoss::Config loss_config;
        loss_config.num_classes = num_classes;
        loss_config.num_boxes = YOLODarknet::kBoxesPerCell;
        loss_config.side = YOLODarknet::kGridSize;
        loss_config.sqrt_wh = true;
        loss_config.rescore = true;

        fs::create_directories(results_dir);
        std::ofstream csv_file((results_dir / "metrics_custom.csv").string());
        csv_file << "Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;mAP@0.5\n";

        for (int epoch = 1; epoch <= total_epochs; ++epoch)
        {
            auto epoch_start_time = std::chrono::steady_clock::now();
            const float current_lr = scheduled_learning_rate(config, epoch);

            for (auto& layer : custom_model.get_all_layers())
            {
                layer->learning_rate = current_lr;
                layer->momentum = momentum;
                if (layer->frozen())
                {
                    layer->eval();
                }
                else
                {
                    layer->train();
                }
            }

            float epoch_train_loss = 0.0F;
            int train_batches = 0;

            train_loader.reset();
            dl::UniqueCudaStream copy_streams[2];
            std::optional<Batch> batches[2];
            bool has_batch[2] { false, false };

            if (train_loader.has_next())
            {
                batches[0] = train_loader.get_batch(copy_streams[0].get());
                has_batch[0] = true;
            }

            int slot = 0;
            while (has_batch[slot])
            {
                const int next = 1 - slot;
                const cudaStream_t compute_stream = copy_streams[slot].get();
                CHECK_CUDA(cudaStreamSynchronize(compute_stream));

                const dl::StreamGuard stream_guard(compute_stream);
                dl::Tensor pred = custom_model.forward(batches[slot]->images, compute_stream);

                const float batch_loss =
                    DarknetDetectionLoss::loss(batches[slot]->targets, pred, loss_config, compute_stream)
                        .to_host(compute_stream)
                        .front();

                dl::Tensor grad_error = trainer.clip_loss_gradient(
                    DarknetDetectionLoss::loss_derivative(batches[slot]->targets, pred, loss_config, compute_stream));

                auto layers = custom_model.get_all_layers();
                const int first_trainable = first_unfrozen_index(layers);
                for (int index = static_cast<int>(layers.size()) - 1; index >= first_trainable; --index)
                {
                    grad_error = layers[static_cast<std::size_t>(index)]->backward(grad_error, compute_stream);
                }
                trainer.clip_parameter_gradients(compute_stream);
                for (auto& layer : layers)
                {
                    layer->step(compute_stream);
                }

                epoch_train_loss += batch_loss;
                train_batches++;
                if (train_batches == 1 || train_batches % 50 == 0)
                {
                    LOG_DEBUG("VOC Darknet epoch {} batch {} last_loss={:.4f} pred {}", epoch, train_batches,
                        batch_loss, pred.describe());
                }

                if (train_loader.has_next())
                {
                    CHECK_CUDA(cudaStreamSynchronize(copy_streams[next].get()));
                    batches[next] = train_loader.get_batch(copy_streams[next].get());
                    has_batch[next] = true;
                }
                else
                {
                    has_batch[next] = false;
                    batches[next].reset();
                }
                slot = next;
            }
            CHECK_CUDA(cudaStreamSynchronize(copy_streams[0].get()));
            CHECK_CUDA(cudaStreamSynchronize(copy_streams[1].get()));
            const float avg_train_loss = epoch_train_loss / static_cast<float>(std::max(1, train_batches));

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
                epoch_test_loss += DarknetDetectionLoss::loss(batch.targets, pred, loss_config).to_host().front();
                test_batches++;

                auto batch_pred = predicted_from_batch(pred, conf_threshold, num_classes, nms_threshold);
                auto batch_gt = truth_from_batch(batch.targets, num_classes);
                predicted_detections.insert(predicted_detections.end(), batch_pred.begin(), batch_pred.end());
                ground_truth_detections.insert(ground_truth_detections.end(), batch_gt.begin(), batch_gt.end());
            }
            const float avg_test_loss = epoch_test_loss / static_cast<float>(std::max(1, test_batches));
            const float map50 = mean_average_precision(predicted_detections, ground_truth_detections, 0.5F);

            auto epoch_end_time = std::chrono::steady_clock::now();
            auto epoch_duration =
                std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time).count();

            const auto vram = current_vram_mib();
            log_train_epoch("VOC Darknet", epoch, total_epochs, avg_train_loss, avg_test_loss, epoch_duration, vram);
            LOG_INFO("VOC Darknet | mAP@0.5: {}", map50);
            LOG_FLUSH();

            csv_file << epoch << ";" << avg_train_loss << ";" << avg_test_loss << ";" << epoch_duration << ";" << vram
                     << ";" << map50 << "\n";
            csv_file.flush();
        }

        const std::string save_path = (results_dir / "yolov1_voc_darknet_final.pt").string();
        trainer.save(save_path);
        LOG_INFO("Final Darknet-faithful model saved: {}", save_path);
        LOG_FLUSH();
        return 0;
    }
    catch (const std::exception& exception)
    {
        LOG_ERROR("VOC Darknet pipeline failed: {}", exception.what());
        LOG_FLUSH();
        return 1;
    }
}
