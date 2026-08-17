#include "experiment_config.hpp"
#include "image_inference.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/utils.hpp"
#include "TorchYOLO.hpp"

#include <cstring>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

auto torch_to_host(const torch::Tensor& tensor) -> std::vector<float>
{
    const auto cpu = tensor.contiguous().to(torch::kCPU).to(torch::kFloat32);
    std::vector<float> host(static_cast<std::size_t>(cpu.numel()));
    std::memcpy(host.data(), cpu.data_ptr<float>(), host.size() * sizeof(float));
    return host;
}

int main(int argc, char* argv[])
{
    const nlohmann::json config = load_pipeline_config("voc_torch");
    const int num_classes = config.value("num_classes", 20);
    const float conf_threshold = config.value("conf_threshold", 0.2F);
    const float nms_threshold = config.value("nms_threshold", 0.5F);
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/voc"));
    const fs::path default_model = results_dir / "yolov1_voc_torch_final.pt";
    const fs::path default_images = resolve_from_source(config.value("dataset_root", "data/VOCdevkit")) / "VOC2012" / "JPEGImages";

    if (argc != 1 && argc != 3)
    {
        LOG_ERROR("Usage: {} [<model_path.pt> <image_path_or_dir>]", argv[0]);
        return 1;
    }

    const fs::path model_path = (argc == 3) ? fs::path(argv[1]) : default_model;
    const fs::path image_path = (argc == 3) ? fs::path(argv[2]) : default_images;
    const fs::path out_dir = results_dir / "predictions_torch";
    fs::create_directories(out_dir);

    if (!fs::exists(model_path))
    {
        LOG_ERROR("Torch VOC weights not found: {}", model_path.string());
        return 1;
    }

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    YOLOv1 torch_model(num_classes);
    torch::load(torch_model, model_path.string());
    torch_model->to(device);
    torch_model->eval();

    const auto images = collect_image_paths(image_path);
    if (images.empty())
    {
        LOG_ERROR("No images found at {}", image_path.string());
        return 1;
    }

    LOG_INFO("[VOC TORCH INFERENCE] model={} images={}", model_path.string(), images.size());
    for (const auto& image_file : images)
    {
        cv::Mat img = cv::imread(image_file.string());
        if (img.empty())
        {
            LOG_ERROR("Failed to load image: {}", image_file.string());
            continue;
        }
        auto prepared = prepare_yolo_input(img, 448);
        auto input = torch::from_blob(prepared.second.data(), { 1, 3, 448, 448 }, torch::kFloat32).clone().to(device);
        torch::Tensor output;
        {
            torch::NoGradGuard no_grad;
            output = torch_model->forward(input);
        }
        auto raw_detections = decode_yolo_tensor(torch_to_host(output), conf_threshold, img.cols, img.rows, num_classes);
        auto final_detections = apply_nms(raw_detections, nms_threshold);
        draw_detections(img, final_detections, VOC_CLASSES, cv::Scalar(0, 255, 0));
        const std::string save_path = (out_dir / ("inference_" + image_file.filename().string())).string();
        cv::imwrite(save_path, img);
        LOG_INFO("Image ({} clean detections) saved at: {}", final_detections.size(), save_path);
    }
    return 0;
}
