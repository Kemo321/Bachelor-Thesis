#include "experiment_config.hpp"
#include "image_inference.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/utils.hpp"
#include "YOLO.hpp"

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Synthetic-dataset class indices in the last dimension of the YOLO grid.
const std::vector<std::string> SYNTH_CLASSES = { "square", "circle", "triangle" };

int main()
{
    const nlohmann::json config = load_pipeline_config("synthetic_custom");
    const int num_classes = config.value("num_classes", 3);
    // The confidence threshold drops weak boxes at decode time; the NMS threshold is the maximum IoU of overlapping boxes.
    const float conf_threshold = config.value("conf_threshold", 0.10F);
    const float nms_threshold = config.value("nms_threshold", 0.45F);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/Synthetic3/train"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/synthetic"));
    // Keep the console log next to the CSV. A new run replaces this file.
    open_results_log(results_dir, "log_infer_custom.txt");
    const fs::path model_path = results_dir / "yolov1_synthetic_custom_final.pt";
    const fs::path out_dir = results_dir / "predictions_custom";
    fs::create_directories(out_dir);

    if (!fs::exists(model_path))
    {
        LOG_ERROR("Custom Synthetic weights not found: {}", model_path.string());
        return 1;
    }

    // Load weights into the YOLO layers and switch them to GPU eval mode.
    YOLO custom_model(num_classes);
    Network custom_net(custom_model.get_all_layers(), 0.0F);
    custom_net.load(model_path.string());
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->eval();
    }

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root.string(), train_paths, val_paths, test_paths, SYNTH_CLASSES);
    std::vector<std::string> sample_images = test_paths.images.empty() ? train_paths.images : test_paths.images;
    if (sample_images.empty())
    {
        LOG_ERROR("No Synthetic images found under {}", data_root.string());
        return 1;
    }
    std::mt19937 rng { std::random_device {}() };
    std::shuffle(sample_images.begin(), sample_images.end(), rng);
    sample_images.resize(std::min<std::size_t>(30, sample_images.size()));
    log_inference_start("Synth Custom Infer", "custom", model_path.string(), sample_images.size(), conf_threshold,
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
        const dl::Tensor input = dl::Tensor::from_host({ 1, 3, 448, 448 }, prepared.second, dl::Device::GPU);
        const std::vector<float> output = custom_model.forward(input).to_host();
        auto raw_detections = decode_yolo_tensor(output, conf_threshold, image.cols, image.rows, num_classes);
        // NMS keeps boxes whose IoU does not exceed the threshold.
        auto kept_detections = apply_nms(raw_detections, nms_threshold);
        // Draw only after the confidence threshold and NMS. This path does not compute mAP.
        draw_detections(image, kept_detections, SYNTH_CLASSES, cv::Scalar(0, 0, 255));
        const std::string filename = fs::path(img_path).filename().string();
        // Save the image with red boxes as custom_<file>.
        cv::imwrite((out_dir / ("custom_" + filename)).string(), image);
        ++saved;
    }
    log_inference_done("Synth Custom Infer", saved, out_dir.string());
    return 0;
}
