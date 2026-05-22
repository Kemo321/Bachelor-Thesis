#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "DeepLearnLib/TorchYOLO.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/utils.hpp"

namespace fs = std::filesystem;

const std::vector<std::string> SYNTH_CLASSES = { "square", "circle", "triangle" };

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "==========================================\n";
    std::cout << "[SYNTHETIC INFERENCE] Starting inference...\n";
    std::cout << "==========================================\n";

    const std::string data_root = "../../data/Synthetic3/train";
    const std::string results_dir = "../../results/synthetic";
    const std::string out_dir = results_dir + "/comparisons";
    fs::create_directories(out_dir);

    YOLOv1 torch_model(3);
    std::string torch_path = results_dir + "/yolov1_synthetic_torch_final.pt";
    if(fs::exists(torch_path)) {
        torch::load(torch_model, torch_path);
        torch_model->to(device);
        torch_model->eval();
        std::cout << "[OK] Torch model loaded.\n";
    } else {
        std::cerr << "[ERROR] Not found: " << torch_path << "\n"; return -1;
    }

    YOLO custom_model(3);
    Network custom_net(custom_model->get_all_layers(), 0.0f);
    std::string custom_path = results_dir + "/yolov1_synthetic_custom_final.pt";
    if(fs::exists(custom_path)) {
        custom_net.load(custom_path);
        for(auto& l : custom_model->get_all_layers()) { l->to(device); l->eval(); }
        std::cout << "[OK] Custom model loaded.\n";
    } else {
        std::cerr << "[ERROR] Not found: " << custom_path << "\n"; return -1;
    }

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root, train_paths, val_paths, test_paths, SYNTH_CLASSES);

    std::vector<std::string> sample_images = test_paths.images.empty() ? train_paths.images : test_paths.images;
    if (sample_images.empty()) {
        std::cerr << "[ERROR] No data!\n"; return -1;
    }

    std::random_device rd; std::mt19937 g(rd());
    std::shuffle(sample_images.begin(), sample_images.end(), g);
    size_t images_to_process = std::min<size_t>(30, sample_images.size());

    for (size_t idx = 0; idx < images_to_process; ++idx) {
        std::string img_path = sample_images[idx];
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;

        cv::Mat resized; cv::resize(img, resized, cv::Size(448, 448));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
        auto input = torch::from_blob(resized.data, {1, 448, 448, 3}, torch::kFloat32).permute({0, 3, 1, 2}).contiguous().to(device);

        fs::path p(img_path);
        std::string filename = p.filename().string();

        torch::Tensor out_torch;
        { torch::NoGradGuard no_grad; out_torch = torch_model->forward(input).cpu().view({1, 7, 7, 13}); }
        
        cv::Mat img_torch = img.clone();
        auto raw_torch = decode_yolo_tensor(out_torch, 0.10f, img.cols, img.rows, 3);
        auto final_torch = apply_nms(raw_torch, 0.45f);
        draw_detections(img_torch, final_torch, SYNTH_CLASSES); 
        cv::imwrite(out_dir + "/torch_" + filename, img_torch);

        torch::Tensor out_custom;
        { torch::NoGradGuard no_grad; out_custom = custom_model->forward(input).cpu().view({1, 7, 7, 13}); }
        
        cv::Mat img_custom = img.clone();
        auto raw_custom = decode_yolo_tensor(out_custom, 0.10f, img.cols, img.rows, 3);
        auto final_custom = apply_nms(raw_custom, 0.45f);
        draw_detections(img_custom, final_custom, SYNTH_CLASSES); 
        cv::imwrite(out_dir + "/custom_" + filename, img_custom);
        
        std::cout << "Generated pairs for: " << filename << "\n";
    }

    std::cout << "[SUCCESS] Comparison completed. Images are located in " << out_dir << "\n";
    return 0;
}