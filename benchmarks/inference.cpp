#include <iostream>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "DeepLearnLib/TorchYOLO.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/utils.hpp"

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ./inference <--torch|--custom> <model_path.pt> <image_path_or_dir>\n";
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
    if (fs::is_directory(input_path)) {
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (!entry.is_regular_file()) continue;
            const std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                images.push_back(entry.path());
            }
        }
        std::sort(images.begin(), images.end());
        if (images.size() > 50) images.resize(50);
    } else {
        images.push_back(input_path);
    }

    if (images.empty()) {
        std::cerr << "[ERROR] No images found!\n";
        return -1;
    }

    YOLOv1 torch_model(20);
    YOLO custom_model(20);
    if (mode == "--torch") {
        torch::load(torch_model, model_path);
        torch_model->to(device);
        torch_model->eval();
    } else {
        Network network(custom_model->get_all_layers(), 0.0f);
        network.load(model_path);
        for (auto& l : custom_model->get_all_layers()) { l->to(device); l->eval(); }
    }

    for (const auto& image_file : images) {
        cv::Mat img = cv::imread(image_file.string());
        if (img.empty()) {
            std::cerr << "[ERROR] Failed to load image: " << image_file.string() << "\n";
            continue;
        }

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(448, 448));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
        auto input = torch::from_blob(resized.data, {1, 448, 448, 3}, torch::kFloat32).permute({0, 3, 1, 2}).contiguous().to(device);

        torch::Tensor output;
        if (mode == "--torch") {
            { torch::NoGradGuard no_grad; output = torch_model->forward(input).cpu().view({1, 7, 7, 30}); }
        } else {
            { torch::NoGradGuard no_grad; output = custom_model->forward(input).cpu().view({1, 7, 7, 30}); }
        }

        auto raw_detections = decode_yolo_tensor(output, conf_threshold, img.cols, img.rows, 20);
        auto final_detections = apply_nms(raw_detections, nms_threshold);

        cv::Scalar box_color = (mode == "--torch") ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        draw_detections(img, final_detections, VOC_CLASSES, box_color);

        std::string save_path = out_dir + "/inference_" + image_file.filename().string();
        cv::imwrite(save_path, img);
        std::cout << "[INFO] Image (" << final_detections.size() << " clean detections) saved at: " << save_path << "\n";
    }

    return 0;
}