#include "experiment_config.hpp"
#include "image_inference.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "YOLO.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/utils.hpp"

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> BCCD_CLASSES = { "RBC", "WBC", "Platelets" };

int main()
{
    const nlohmann::json config = load_pipeline_config("bccd_custom");
    const int num_classes = config.value("num_classes", 3);
    const float conf_threshold = config.value("conf_threshold", 0.15F);
    const float nms_threshold = config.value("nms_threshold", 0.60F);
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/BCCD_Dataset/BCCD"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/bccd"));
    const fs::path model_path = results_dir / "yolov1_bccd_custom_final.pt";
    const fs::path out_dir = results_dir / "predictions_custom";
    fs::create_directories(out_dir);

    if (!fs::exists(model_path))
    {
        LOG_ERROR("Custom BCCD weights not found: {}", model_path.string());
        return 1;
    }

    YOLO custom_model(num_classes);
    Network custom_net(custom_model.get_all_layers(), 0.0F);
    custom_net.load(model_path.string());
    for (auto& layer : custom_model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->eval();
    }
    LOG_INFO("[BCCD CUSTOM INFERENCE] Loaded {}", model_path.string());

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

    for (const auto& img_path : sample_images)
    {
        cv::Mat img = cv::imread(img_path);
        if (img.empty())
        {
            continue;
        }
        auto prepared = prepare_yolo_input(img, 448);
        const dl::Tensor input = dl::Tensor::from_host({ 1, 3, 448, 448 }, prepared.second, dl::Device::GPU);
        const std::vector<float> output = custom_model.forward(input).to_host();
        auto raw = decode_yolo_tensor(output, conf_threshold, img.cols, img.rows, num_classes);
        auto final_det = apply_nms(raw, nms_threshold);
        draw_detections(img, final_det, BCCD_CLASSES, cv::Scalar(0, 0, 255));
        const std::string filename = fs::path(img_path).filename().string();
        cv::imwrite((out_dir / ("custom_" + filename)).string(), img);
        LOG_INFO("Wrote {}", (out_dir / ("custom_" + filename)).string());
    }
    return 0;
}
