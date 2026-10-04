#include "experiment_config.hpp"
#include "image_inference.hpp"
#include "run_metrics.hpp"

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

// BCCD class indices in the last dimension of the YOLO grid.
const std::vector<std::string> BCCD_CLASSES = { "RBC", "WBC", "Platelets" };

// Copy the tensor to the host as a float vector, the shared layout for the YOLO decoder.
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
    // The confidence threshold drops weak boxes at decode time; the NMS threshold is the maximum IoU of overlapping boxes.
    const float conf_threshold = config.value("conf_threshold", 0.15F);
    const float nms_threshold = config.value("nms_threshold", 0.60F);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/BCCD_Dataset/BCCD"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/bccd"));
    // Keep the console log next to the CSV. A new run replaces this file.
    open_results_log(results_dir, "log_infer_torch.txt");
    const fs::path model_path = results_dir / "yolov1_bccd_torch_final.pt";
    const fs::path out_dir = results_dir / "predictions_torch";
    fs::create_directories(out_dir);

    if (!fs::exists(model_path))
    {
        LOG_ERROR("Torch BCCD weights not found: {}", model_path.string());
        return 1;
    }

    // Load the LibTorch weights and evaluate on the device selected by torch::cuda::is_available().
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    YOLOv1 torch_model(num_classes);
    torch::load(torch_model, model_path.string());
    torch_model->to(device);
    torch_model->eval();

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
    log_inference_start("BCCD Torch Infer", "torch", model_path.string(), sample_images.size(), conf_threshold,
        nms_threshold, out_dir.string());

    std::size_t saved = 0;
    for (const auto& img_path : sample_images)
    {
        cv::Mat image = cv::imread(img_path);
        if (image.empty())
        {
            continue;
        }
        auto prepared = prepare_yolo_input(image, 448);
        auto input = torch::from_blob(prepared.second.data(), { 1, 3, 448, 448 }, torch::kFloat32).clone().to(device);
        torch::Tensor output;
        {
            torch::NoGradGuard no_grad;
            output = torch_model->forward(input);
        }
        auto raw_detections = decode_yolo_tensor(torch_to_host(output), conf_threshold, image.cols, image.rows, num_classes);
        // NMS keeps boxes whose IoU does not exceed the threshold.
        auto kept_detections = apply_nms(raw_detections, nms_threshold);
        // Draw only after the confidence threshold and NMS. This path does not compute mAP.
        draw_detections(image, kept_detections, BCCD_CLASSES, cv::Scalar(0, 255, 0));
        const std::string filename = fs::path(img_path).filename().string();
        // Save the image with green boxes as torch_<file>.
        cv::imwrite((out_dir / ("torch_" + filename)).string(), image);
        ++saved;
    }
    log_inference_done("BCCD Torch Infer", saved, out_dir.string());
    return 0;
}
