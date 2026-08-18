#pragma once

#include "classification_eval.hpp"

#include "DeepLearnLib/Logger.hpp"

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

inline auto write_nchw_png(const std::filesystem::path& path, const std::vector<float>& nchw, int channels, int height,
    int width) -> void
{
    std::filesystem::create_directories(path.parent_path());
    cv::Mat image(height, width, CV_8UC3);
    const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    for (int row = 0; row < height; ++row)
    {
        auto* pixel = image.ptr<unsigned char>(row);
        for (int col = 0; col < width; ++col)
        {
            const std::size_t spatial = (static_cast<std::size_t>(row) * static_cast<std::size_t>(width))
                + static_cast<std::size_t>(col);
            const float r = (channels >= 3) ? nchw[spatial] : nchw[spatial];
            const float g = (channels >= 3) ? nchw[plane + spatial] : nchw[spatial];
            const float b = (channels >= 3) ? nchw[(2 * plane) + spatial] : nchw[spatial];
            pixel[0] = static_cast<unsigned char>(std::clamp(b * 255.0F, 0.0F, 255.0F));
            pixel[1] = static_cast<unsigned char>(std::clamp(g * 255.0F, 0.0F, 255.0F));
            pixel[2] = static_cast<unsigned char>(std::clamp(r * 255.0F, 0.0F, 255.0F));
            pixel += 3;
        }
    }
    if (!cv::imwrite(path.string(), image))
    {
        throw std::runtime_error("cv::imwrite failed: " + path.string());
    }
}

inline auto write_classification_samples(const std::filesystem::path& directory,
    const std::vector<SamplePrediction>& samples, const std::vector<std::string>& class_names) -> void
{
    std::filesystem::create_directories(directory);
    write_predictions_csv(directory / "predictions.csv", samples, class_names);
    int written = 0;
    for (const auto& sample : samples)
    {
        const std::string truth_name = (sample.truth < static_cast<int>(class_names.size()))
            ? class_names[static_cast<std::size_t>(sample.truth)]
            : std::to_string(sample.truth);
        const std::string pred_name = (sample.pred < static_cast<int>(class_names.size()))
            ? class_names[static_cast<std::size_t>(sample.pred)]
            : std::to_string(sample.pred);
        const std::string filename = (sample.truth == sample.pred ? "ok_" : "err_")
            + std::to_string(sample.index) + "_t" + truth_name + "_p" + pred_name + ".png";
        write_nchw_png(directory / filename, sample.image_nchw, sample.channels, sample.height, sample.width);
        ++written;
    }
    LOG_INFO("Wrote {} classification sample images to {}", written, directory.string());
}
