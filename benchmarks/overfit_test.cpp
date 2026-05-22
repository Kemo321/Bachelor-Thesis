#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "DeepLearnLib/TorchYOLO.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/utils.hpp"

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./overfit_test <--torch|--custom>\n";
        return -1;
    }

    std::string mode = argv[1];
    if (mode != "--torch" && mode != "--custom") {
        std::cerr << "[ERROR] Unknown mode: " << mode << "\n";
        return -1;
    }

    const int batch_size = 8;
    const int total_epochs = 300;
    const std::string data_root = "../../data/VOCdevkit";
    const std::string results_dir = "../../results";
    const float learning_rate = 2e-5F;

    std::cout << "========================================\n";
    std::cout << "[OVERFIT TEST] Mode: " << mode << " | Batch: " << batch_size << "\n";
    std::cout << "========================================\n";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root + "/VOC2012", train_paths, val_paths, test_paths, VOC_CLASSES);

    if (train_paths.images.empty()) {
        std::cerr << "[ERROR] No data in the data folder!\n"; return -1;
    }

    DataPaths tiny_paths;
    for(int i = 0; i < batch_size && i < train_paths.images.size(); ++i) {
        tiny_paths.images.push_back(train_paths.images[i]);
        tiny_paths.labels.push_back(train_paths.labels[i]);
    }

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(tiny_paths, false, VOC_CLASSES).map(torch::data::transforms::Stack<>()), 
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    if (mode == "--torch") {
        YOLOv1 model(20);
        model->to(device);
        model->train();
        
        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9));

        for (int epoch = 1; epoch <= total_epochs; ++epoch) {
            float epoch_loss = 0.0F;
            for (auto& batch : *train_loader) {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);

                optimizer.zero_grad();
                auto pred = model->forward(data);
                auto loss = YOLOLoss::loss(target, pred, 20);

                loss.backward();
                optimizer.step();

                epoch_loss += loss.item().toFloat();
            }
            if (epoch % 10 == 0 || epoch == 1) {
                std::cout << "Epoch [" << std::setw(3) << epoch << "/" << total_epochs << "] Loss: " << epoch_loss << "\n";
            }
        }
        std::string save_path = results_dir + "/yolov1_torch_overfitted.pt";
        torch::save(model, save_path);
        std::cout << "\n[INFO] Torch model saved: " << save_path << "\n";

        model->eval();
        std::string drawn_dir = results_dir + "/overfit_drawn_torch";
        fs::create_directories(drawn_dir);

        for (const auto& img_path : tiny_paths.images) {
            cv::Mat img = cv::imread(img_path);
            if (img.empty()) continue;
            cv::Mat resized; cv::resize(img, resized, cv::Size(448, 448));
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
            resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
            auto input = torch::from_blob(resized.data, {1, 448, 448, 3}, torch::kFloat32).permute({0, 3, 1, 2}).contiguous().to(device);
            
            torch::Tensor output;
            { torch::NoGradGuard no_grad; output = model->forward(input).cpu().view({1, 7, 7, 30}); }
            
            auto raw_det = decode_yolo_tensor(output, 0.10f, img.cols, img.rows, 20);
            auto final_det = apply_nms(raw_det, 0.45f);
            draw_detections(img, final_det, VOC_CLASSES, cv::Scalar(0, 255, 0));
            
            fs::path p(img_path);
            cv::imwrite(drawn_dir + "/" + p.filename().string(), img);
        }
        std::cout << "[SUCCESS] Torch-generated images saved in: " << drawn_dir << "\n";

    } else {
        YOLO custom_model(20);
        for (auto& l : custom_model->get_all_layers()) { 
            l->to(device); l->train(); l->learning_rate = learning_rate; 
        }

        for (int epoch = 1; epoch <= total_epochs; ++epoch) {
            float epoch_loss = 0.0F;
            for (auto& batch : *train_loader) {
                auto data = batch.data.to(device); 
                auto target = batch.target.to(device);

                auto pred = custom_model->forward(data);
                auto loss = YOLOLoss::loss(target, pred, 20);
                epoch_loss += loss.item().toFloat();

                auto grad_error = YOLOLoss::loss_derivative(target, pred, 20).clamp(-10.0, 10.0);
                auto layers = custom_model->get_all_layers();
                
                for (auto it = layers.rbegin(); it != layers.rend(); ++it) { 
                    grad_error = (*it)->backward(grad_error); 
                }
                for (auto& l : layers) l->step();
            }
            if (epoch % 10 == 0 || epoch == 1) {
                std::cout << "Epoch [" << std::setw(3) << epoch << "/" << total_epochs << "] Loss: " << epoch_loss << "\n";
            }
        }
        Network trainer(custom_model->get_all_layers(), learning_rate);
        std::string save_path = results_dir + "/yolov1_custom_overfitted.pt";
        trainer.save(save_path);
        std::cout << "\n[INFO] Custom model saved: " << save_path << "\n";

        for (auto& l : custom_model->get_all_layers()) { l->eval(); }
        std::string drawn_dir = results_dir + "/overfit_drawn_custom";
        fs::create_directories(drawn_dir);

        for (const auto& img_path : tiny_paths.images) {
            cv::Mat img = cv::imread(img_path);
            if (img.empty()) continue;
            cv::Mat resized; cv::resize(img, resized, cv::Size(448, 448));
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
            resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
            auto input = torch::from_blob(resized.data, {1, 448, 448, 3}, torch::kFloat32).permute({0, 3, 1, 2}).contiguous().to(device);
            
            torch::Tensor output;
            { torch::NoGradGuard no_grad; output = custom_model->forward(input).cpu().view({1, 7, 7, 30}); }
            
            auto raw_det = decode_yolo_tensor(output, 0.10f, img.cols, img.rows, 20);
            auto final_det = apply_nms(raw_det, 0.45f);
            draw_detections(img, final_det, VOC_CLASSES, cv::Scalar(0, 0, 255));
            
            fs::path p(img_path);
            cv::imwrite(drawn_dir + "/" + p.filename().string(), img);
        }
        std::cout << "[SUCCESS] Custom-generated images saved in: " << drawn_dir << "\n";
    }

    return 0;
}