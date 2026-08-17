#include "experiment_config.hpp"
#include "image_inference.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/utils.hpp"
#include "TorchYOLO.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> BCCD_CLASSES = { "RBC", "WBC", "Platelets" };

auto torch_to_host(const torch::Tensor& tensor) -> std::vector<float>
{
    const auto cpu = tensor.contiguous().to(torch::kCPU).to(torch::kFloat32);
    std::vector<float> host(static_cast<std::size_t>(cpu.numel()));
    std::memcpy(host.data(), cpu.data_ptr<float>(), host.size() * sizeof(float));
    return host;
}

int main()
{
    const nlohmann::json config = load_pipeline_config("bccd_torch");
    const int num_classes = config.value("num_classes", 3);
    const float conf_threshold = config.value("conf_threshold", 0.15F);
    const float nms_threshold = config.value("nms_threshold", 0.60F);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/BCCD_Dataset/BCCD"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/bccd"));
    const fs::path model_path = results_dir / "yolov1_bccd_torch_final.pt";
    const fs::path out_dir = results_dir / "predictions_torch";
    fs::create_directories(out_dir);

    if (!fs::exists(model_path))
    {
        LOG_ERROR("Torch BCCD weights not found: {}", model_path.string());
        return 1;
    }

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    YOLOv1 torch_model(num_classes);
    torch::load(torch_model, model_path.string());
    torch_model->to(device);
    torch_model->eval();
    LOG_INFO("[BCCD TORCH INFERENCE] Loaded {}", model_path.string());

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root.string(), train_paths, val_paths, test_paths, BCCD_CLASSES);
    std::vector<std::string> sample_images = test_paths.images.empty() ? train_paths.images : test_paths.images;
    if (sample_images.empty())
    {
        LOG_ERROR("No BCCD images found under {}", data_root.string());
        return 1;
    }
    std::mt19937 rng { std::random_device {}() };
    std::shuffle(sample_images.begin(), sample_images.end(), rng);
    sample_images.resize(std::min<std::size_t>(30, sample_images.size()));

    std::size_t saved = 0;
    for (const auto& img_path : sample_images)
    {
        cv::Mat img = cv::imread(img_path);
        if (img.empty())
        {
            continue;
        }
        auto prepared = prepare_yolo_input(img, 448);
        auto input = torch::from_blob(prepared.second.data(), { 1, 3, 448, 448 }, torch::kFloat32).clone().to(device);
        torch::Tensor output;
        {
            torch::NoGradGuard no_grad;
            output = torch_model->forward(input);
        }
        auto raw = decode_yolo_tensor(torch_to_host(output), conf_threshold, img.cols, img.rows, num_classes);
        auto final_det = apply_nms(raw, nms_threshold);
        draw_detections(img, final_det, BCCD_CLASSES, cv::Scalar(0, 255, 0));
        const std::string filename = fs::path(img_path).filename().string();
        cv::imwrite((out_dir / ("torch_" + filename)).string(), img);
        ++saved;
    }
    LOG_INFO("Successfully processed and saved {} images.", saved);
    return 0;
}
