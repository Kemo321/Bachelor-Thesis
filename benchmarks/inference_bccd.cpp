#include "DeepLearnLib/Logger.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/utils.hpp"
#include "TorchYOLO.hpp"

namespace fs = std::filesystem;

const std::vector<std::string> BCCD_CLASSES = { "RBC", "WBC", "Platelets" };

namespace
{

constexpr int kInputSize = 448;
constexpr int kChannels = 3;
constexpr float kConfThreshold = 0.15F;
constexpr float kNmsThreshold = 0.60F;

auto hwc_to_nchw(const cv::Mat& image) -> std::vector<float>
{
    const int plane = kInputSize * kInputSize;
    std::vector<float> nchw(static_cast<std::size_t>(kChannels * plane));
    for (int row = 0; row < kInputSize; ++row)
    {
        const auto* pixel = image.ptr<float>(row);
        for (int col = 0; col < kInputSize; ++col)
        {
            const int spatial = (row * kInputSize) + col;
            nchw[static_cast<std::size_t>(spatial)] = pixel[0];
            nchw[static_cast<std::size_t>(plane + spatial)] = pixel[1];
            nchw[static_cast<std::size_t>((2 * plane) + spatial)] = pixel[2];
            pixel += 3;
        }
    }
    return nchw;
}

auto torch_to_host(const torch::Tensor& tensor) -> std::vector<float>
{
    const auto cpu = tensor.contiguous().to(torch::kCPU).to(torch::kFloat32);
    std::vector<float> host(static_cast<std::size_t>(cpu.numel()));
    std::memcpy(host.data(), cpu.data_ptr<float>(), host.size() * sizeof(float));
    return host;
}

} // namespace

int main()
{
    torch::Device torch_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    LOG_INFO("==========================================");
    LOG_INFO("[BCCD INFERENCE] Starting inference...");
    LOG_INFO("==========================================");

    const std::string data_root = "../../data/BCCD_Dataset/BCCD";
    const std::string results_dir = "../../results/bccd";
    const std::string out_dir = results_dir + "/comparisons";
    fs::create_directories(out_dir);

    const std::string torch_path = results_dir + "/yolov1_bccd_torch_final.pt";
    const std::string custom_path = results_dir + "/yolov1_bccd_custom_final.pt";
    const bool have_torch = fs::exists(torch_path);
    const bool have_custom = fs::exists(custom_path);
    if (!have_torch && !have_custom)
    {
        LOG_ERROR("No BCCD models found under {}", results_dir);
        return -1;
    }

    YOLOv1 torch_model(3);
    if (have_torch)
    {
        torch::load(torch_model, torch_path);
        torch_model->to(torch_device);
        torch_model->eval();
        LOG_INFO("[OK] Torch model loaded.");
    }
    else
    {
        LOG_WARN("Skipping Torch inference; missing {}", torch_path);
    }

    YOLO custom_model(3);
    if (have_custom)
    {
        Network custom_net(custom_model.get_all_layers(), 0.0F);
        custom_net.load(custom_path);
        for (auto& layer : custom_model.get_all_layers())
        {
            layer->to(dl::Device::GPU);
            layer->eval();
        }
        LOG_INFO("[OK] Custom model loaded.");
    }
    else
    {
        LOG_WARN("Skipping custom inference; missing {}", custom_path);
    }

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root, train_paths, val_paths, test_paths, BCCD_CLASSES);

    if (test_paths.images.empty())
    {
        LOG_ERROR("No test data found!");
        return -1;
    }

    std::vector<std::string> sample_images = test_paths.images;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(sample_images.begin(), sample_images.end(), rng);
    const std::size_t images_to_process = std::min<std::size_t>(30, sample_images.size());

    for (std::size_t idx = 0; idx < images_to_process; ++idx)
    {
        const std::string img_path = sample_images[idx];
        cv::Mat img = cv::imread(img_path);
        if (img.empty())
        {
            continue;
        }

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(kInputSize, kInputSize));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0F / 255.0F);
        std::vector<float> nchw = hwc_to_nchw(resized);
        const std::string filename = fs::path(img_path).filename().string();

        if (have_torch)
        {
            auto input = torch::from_blob(nchw.data(), { 1, kChannels, kInputSize, kInputSize }, torch::kFloat32)
                             .clone()
                             .to(torch_device);
            torch::Tensor out_torch;
            {
                torch::NoGradGuard no_grad;
                out_torch = torch_model->forward(input);
            }
            cv::Mat img_torch = img.clone();
            auto raw_torch = decode_yolo_tensor(torch_to_host(out_torch), kConfThreshold, img.cols, img.rows, 3);
            auto final_torch = apply_nms(raw_torch, kNmsThreshold);
            draw_detections(img_torch, final_torch, BCCD_CLASSES, cv::Scalar(0, 255, 0));
            cv::imwrite(out_dir + "/torch_" + filename, img_torch);
        }

        if (have_custom)
        {
            const dl::Tensor input = dl::Tensor::from_host({ 1, kChannels, kInputSize, kInputSize }, nchw, dl::Device::GPU);
            const std::vector<float> out_custom = custom_model.forward(input).to_host();
            cv::Mat img_custom = img.clone();
            auto raw_custom = decode_yolo_tensor(out_custom, kConfThreshold, img.cols, img.rows, 3);
            auto final_custom = apply_nms(raw_custom, kNmsThreshold);
            draw_detections(img_custom, final_custom, BCCD_CLASSES, cv::Scalar(0, 0, 255));
            cv::imwrite(out_dir + "/custom_" + filename, img_custom);
        }

        LOG_INFO("Generated pairs for: {}", filename);
    }

    LOG_INFO("[SUCCESS] Comparison completed. Images are located in {}", out_dir);
    return 0;
}
