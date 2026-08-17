#pragma once

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

inline auto hwc_float_to_nchw(const cv::Mat& image, int input_size, int channels) -> std::vector<float>
{
    const int plane = input_size * input_size;
    std::vector<float> nchw(static_cast<std::size_t>(channels * plane));
    for (int row = 0; row < input_size; ++row)
    {
        const auto* pixel = image.ptr<float>(row);
        for (int col = 0; col < input_size; ++col)
        {
            const int spatial = (row * input_size) + col;
            for (int channel = 0; channel < channels; ++channel)
            {
                nchw[static_cast<std::size_t>((channel * plane) + spatial)] = pixel[channel];
            }
            pixel += channels;
        }
    }
    return nchw;
}

inline auto collect_image_paths(const std::filesystem::path& input_path, std::size_t max_images = 50)
    -> std::vector<std::filesystem::path>
{
    std::vector<std::filesystem::path> images;
    if (std::filesystem::is_directory(input_path))
    {
        for (const auto& entry : std::filesystem::directory_iterator(input_path))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            const std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".JPG" || ext == ".JPEG")
            {
                images.push_back(entry.path());
            }
        }
        std::sort(images.begin(), images.end());
        if (images.size() > max_images)
        {
            images.resize(max_images);
        }
    }
    else
    {
        images.push_back(input_path);
    }
    return images;
}

inline auto prepare_yolo_input(const cv::Mat& img, int input_size) -> std::pair<cv::Mat, std::vector<float>>
{
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    return { resized, hwc_float_to_nchw(resized, input_size, 3) };
}
