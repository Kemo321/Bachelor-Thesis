#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/utils.hpp"

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        LOG_ERROR("Usage: ./overfit_test <--torch|--custom>");
        return -1;
    }

    std::string mode = argv[1];
    if (mode != "--torch" && mode != "--custom")
    {
        LOG_ERROR("Unknown mode: {}", mode);
        return -1;
    }

    const int batch_size = 8;
    const int total_epochs = 300;
    const std::string data_root = "../../data/VOCdevkit";
    const std::string results_dir = "../../results";
    const float learning_rate = 2e-5F;

    LOG_INFO("========================================");
    LOG_INFO("[OVERFIT TEST] Mode: {} | Batch: {}", mode, batch_size);
    LOG_INFO("========================================");

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root + "/VOC2012", train_paths, val_paths, test_paths, VOC_CLASSES);

    if (train_paths.images.empty())
    {
        LOG_ERROR("No data in the data folder!");
        return -1;
    }

    DataPaths tiny_paths;
    for (int i = 0; i < batch_size && i < static_cast<int>(train_paths.images.size()); ++i)
    {
        tiny_paths.images.push_back(train_paths.images[i]);
        tiny_paths.labels.push_back(train_paths.labels[i]);
    }

    if (mode == "--torch")
    {
        LOG_ERROR("--torch overfit still expects a LibTorch loss. YOLOLoss is dl::Tensor-only; use --custom.");
        return -1;
    }

    CustomDataLoader train_loader(tiny_paths, batch_size, false, VOC_CLASSES);

    YOLO custom_model(20);
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->train();
        layer->learning_rate = learning_rate;
    }

    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        float epoch_loss = 0.0F;
        train_loader.reset();
        while (train_loader.has_next())
        {
            Batch batch = train_loader.get_batch();

            dl::Tensor pred = custom_model.forward(batch.images);
            epoch_loss += YOLOLoss::loss(batch.targets, pred, 20).to_host().front();

            dl::Tensor grad_error = YOLOLoss::loss_derivative(batch.targets, pred, 20).clamp(-10.0F, 10.0F);
            auto layers = custom_model.get_all_layers();

            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {
                grad_error = (*it)->backward(grad_error);
            }
            for (auto& layer : layers)
            {
                layer->step();
            }
        }
        if (epoch % 10 == 0 || epoch == 1)
        {
            LOG_INFO("Epoch [{}/{}] Loss: {}", epoch, total_epochs, epoch_loss);
        }
    }
    Network trainer(custom_model.get_all_layers(), learning_rate);
    std::string save_path = results_dir + "/yolov1_custom_overfitted.pt";
    trainer.save(save_path);
    LOG_INFO("Custom model saved: {}", save_path);

    for (auto& layer : custom_model.get_all_layers())
    {
        layer->eval();
    }
    std::string drawn_dir = results_dir + "/overfit_drawn_custom";
    fs::create_directories(drawn_dir);

    for (const auto& img_path : tiny_paths.images)
    {
        cv::Mat img = cv::imread(img_path);
        if (img.empty())
            continue;
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(448, 448));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

        constexpr int kHeight = 448;
        constexpr int kWidth = 448;
        constexpr int kChannels = 3;
        std::vector<float> chw_data(static_cast<size_t>(kChannels * kHeight * kWidth));
        for (int row = 0; row < kHeight; ++row)
        {
            const auto* pixel_row = resized.ptr<cv::Vec3f>(row);
            for (int col = 0; col < kWidth; ++col)
            {
                for (int channel = 0; channel < kChannels; ++channel)
                {
                    chw_data[static_cast<size_t>((channel * kHeight * kWidth) + (row * kWidth) + col)] = pixel_row[col][channel];
                }
            }
        }

        dl::Tensor input = dl::Tensor::from_host({ 1, 3, 448, 448 }, chw_data.data());
        std::vector<float> output_data = custom_model.forward(input).to_host();

        auto raw_det = decode_yolo_tensor(output_data, 0.10f, img.cols, img.rows, 20);
        auto final_det = apply_nms(raw_det, 0.45f);
        draw_detections(img, final_det, VOC_CLASSES, cv::Scalar(0, 0, 255));

        fs::path p(img_path);
        cv::imwrite(drawn_dir + "/" + p.filename().string(), img);
    }
    LOG_INFO("Custom-generated images saved in: {}", drawn_dir);

    return 0;
}
