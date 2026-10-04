#include "experiment_config.hpp"
#include "image_inference.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/utils.hpp"
#include "YOLO.hpp"

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Indices of the 20 VOC classes in the last dimension of the YOLO grid.
const std::vector<std::string> VOC_CLASSES = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

int main(int argc, char* argv[])
{
    const nlohmann::json config = load_pipeline_config("voc_custom");
    const int num_classes = config.value("num_classes", 20);
    // The confidence threshold drops weak boxes at decode time; the NMS threshold is the maximum IoU of overlapping boxes.
    const float conf_threshold = config.value("conf_threshold", 0.25F);
    const float nms_threshold = config.value("nms_threshold", 0.5F);
    const std::string voc_subset = config.value("voc_subset", "VOC2012");
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/voc"));
    // Keep the console log next to the CSV. A new run replaces this file.
    open_results_log(results_dir, "log_infer_custom.txt");
    const fs::path default_model = results_dir / "yolov1_voc_custom_final.pt";
    const fs::path default_images = resolve_from_source(config.value("dataset_root", "data/VOCdevkit")) / voc_subset / "JPEGImages";

    if (argc != 1 && argc != 3)
    {
        LOG_ERROR("Usage: {} [<model_path.pt> <image_path_or_dir>]", argv[0]);
        return 1;
    }

    const fs::path model_path = (argc == 3) ? fs::path(argv[1]) : default_model;
    const fs::path image_path = (argc == 3) ? fs::path(argv[2]) : default_images;
    const fs::path out_dir = results_dir / "predictions_custom";
    fs::create_directories(out_dir);

    if (!fs::exists(model_path))
    {
        LOG_ERROR("Custom VOC weights not found: {}", model_path.string());
        return 1;
    }

    // Load weights into the YOLO layers and switch them to GPU eval mode.
    YOLO custom_model(num_classes);
    Network network(custom_model.get_all_layers(), 0.0F);
    network.load(model_path.string());
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->eval();
    }

    // A directory or a single file. For a directory, collect_image_paths keeps at most 50 images.
    const auto images = collect_image_paths(image_path);
    if (images.empty())
    {
        LOG_ERROR("No images found at {}", image_path.string());
        return 1;
    }

    log_inference_start("VOC Custom Infer", "custom", model_path.string(), images.size(), conf_threshold, nms_threshold,
        out_dir.string());
    std::size_t saved = 0;
    for (const auto& image_file : images)
    {
        cv::Mat image = cv::imread(image_file.string());
        if (image.empty())
        {
            LOG_ERROR("Failed to load image: {}", image_file.string());
            continue;
        }
        auto prepared = prepare_yolo_input(image, 448);
        dl::Tensor input = dl::Tensor::from_host({ 1, 3, 448, 448 }, prepared.second.data());
        std::vector<float> output_data = custom_model.forward(input).to_host();
        auto raw_detections = decode_yolo_tensor(output_data, conf_threshold, image.cols, image.rows, num_classes);
        // NMS keeps boxes whose IoU does not exceed the threshold.
        auto final_detections = apply_nms(raw_detections, nms_threshold);
        // Draw only after the confidence threshold and NMS. This path does not compute mAP.
        draw_detections(image, final_detections, VOC_CLASSES, cv::Scalar(0, 0, 255));
        const std::string save_path = (out_dir / ("inference_" + image_file.filename().string())).string();
        // Save the image with red boxes as inference_<file>.
        cv::imwrite(save_path, image);
        ++saved;
    }
    log_inference_done("VOC Custom Infer", saved, out_dir.string());
    return 0;
}
