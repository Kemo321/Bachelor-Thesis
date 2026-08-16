#include "DeepLearnLib/Logger.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/utils.hpp"
#include "TorchYOLO.hpp"

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        LOG_ERROR("Usage: ./inference <--torch|--custom> <model_path.pt> <image_path_or_dir>");
        return -1;
    }

    std::string mode = argv[1];
    std::string model_path = argv[2];
    std::string image_path = argv[3];

    const float conf_threshold = 0.2f;
    const float nms_threshold = 0.5f;

    std::string out_dir = (mode == "--torch") ? "../../results/predictions_torch" : "../../results/predictions_custom";
    fs::create_directories(out_dir);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    std::vector<fs::path> images;
    fs::path input_path(image_path);
    if (fs::is_directory(input_path))
    {
        for (const auto& entry : fs::directory_iterator(input_path))
        {
            if (!entry.is_regular_file())
                continue;
            const std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
            {
                images.push_back(entry.path());
            }
        }
        std::sort(images.begin(), images.end());
        if (images.size() > 50)
            images.resize(50);
    }
    else
    {
        images.push_back(input_path);
    }

    if (images.empty())
    {
        LOG_ERROR("No images found!");
        return -1;
    }

    YOLOv1 torch_model(20);
    std::unique_ptr<YOLO> custom_model;
    if (mode == "--torch")
    {
        torch::load(torch_model, model_path);
        torch_model->to(device);
        torch_model->eval();
    }
    else if (mode == "--custom")
    {
        custom_model = std::make_unique<YOLO>(20);
        Network network(custom_model->get_all_layers(), 0.0f);
        network.load(model_path);
        for (auto& layer : custom_model->get_all_layers())
        {
            layer->to(dl::Device::GPU);
            layer->eval();
        }
    }
    else
    {
        LOG_ERROR("Unknown mode. Use --torch or --custom.");
        return -1;
    }

    for (const auto& image_file : images)
    {
        cv::Mat img = cv::imread(image_file.string());
        if (img.empty())
        {
            LOG_ERROR("Failed to load image: {}", image_file.string());
            continue;
        }

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(448, 448));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

        std::vector<Detection> raw_detections;
        if (mode == "--torch")
        {
            auto input = torch::from_blob(resized.data, { 1, 448, 448, 3 }, torch::kFloat32).permute({ 0, 3, 1, 2 }).contiguous().to(device);
            torch::Tensor output;
            {
                torch::NoGradGuard no_grad;
                output = torch_model->forward(input).cpu().view({ 1, 7, 7, 30 });
            }
            output = output.contiguous();
            const float* output_ptr = output.data_ptr<float>();
            std::vector<float> output_data(output_ptr, output_ptr + output.numel());
            raw_detections = decode_yolo_tensor(output_data, conf_threshold, img.cols, img.rows, 20);
        }
        else
        {
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
            dl::Tensor output = custom_model->forward(input);
            std::vector<float> output_data = output.to_host();
            raw_detections = decode_yolo_tensor(output_data, conf_threshold, img.cols, img.rows, 20);
        }

        auto final_detections = apply_nms(raw_detections, nms_threshold);

        cv::Scalar box_color = (mode == "--torch") ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        draw_detections(img, final_detections, VOC_CLASSES, box_color);

        std::string save_path = out_dir + "/inference_" + image_file.filename().string();
        cv::imwrite(save_path, img);
        LOG_INFO("Image ({} clean detections) saved at: {}", final_detections.size(), save_path);
    }

    return 0;
}
