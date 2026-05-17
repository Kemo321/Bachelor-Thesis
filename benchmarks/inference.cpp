#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "DeepLearnLib/TorchYOLO.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/utils.hpp" // <-- NASZ NOWY MODUŁ NARZĘDZIOWY

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Uzycie: ./inference <--torch|--custom> <sciezka_do_modelu.pt> <sciezka_do_obrazu.jpg>\n";
        return -1;
    }

    std::string mode = argv[1];
    std::string model_path = argv[2];
    std::string image_path = argv[3];
    
    const float conf_threshold = 0.1f;
    const float nms_threshold = 0.5f; 
    
    std::string out_dir = (mode == "--torch") ? "../../results/predictions_torch" : "../../results/predictions_custom";
    fs::create_directories(out_dir);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) { std::cerr << "[BLAD] Nie wczytano obrazu!\n"; return -1; }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(448, 448));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
    auto input = torch::from_blob(resized.data, {1, 448, 448, 3}, torch::kFloat32).permute({0, 3, 1, 2}).contiguous().to(device);

    torch::Tensor output;
    
    if (mode == "--torch") {
        YOLOv1 model(20);
        torch::load(model, model_path);
        model->to(device); model->eval();
        { torch::NoGradGuard no_grad; output = model->forward(input).cpu().view({1, 7, 7, 30}); }
    } else {
        YOLO custom_model(20);
        Network network(custom_model->get_all_layers(), 0.0f);
        network.load(model_path);
        for (auto& l : custom_model->get_all_layers()) { l->to(device); l->eval(); }
        { torch::NoGradGuard no_grad; output = custom_model->forward(input).cpu().view({1, 7, 7, 30}); }
    }

    // ============================================================
    // MAGIA REFAKTORYZACJI: 3 linijki zamiast 50!
    // ============================================================
    auto raw_detections = decode_yolo_tensor(output, conf_threshold, img.cols, img.rows, 20);
    auto final_detections = apply_nms(raw_detections, nms_threshold);
    
    cv::Scalar box_color = (mode == "--torch") ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    draw_detections(img, final_detections, VOC_CLASSES, box_color);
    // ============================================================

    fs::path p(image_path);
    std::string save_path = out_dir + "/inference_" + p.filename().string();
    cv::imwrite(save_path, img);
    std::cout << "[INFO] Obraz (" << final_detections.size() << " czystych detekcji) zapisany w: " << save_path << "\n";
    return 0;
}