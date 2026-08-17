#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

inline auto write_dummy_csv(const std::filesystem::path& csv_path, int num_samples, int num_features, int num_classes,
    unsigned seed) -> void
{
    std::filesystem::create_directories(csv_path.parent_path());
    std::ofstream stream(csv_path);
    if (!stream)
    {
        throw std::runtime_error("Could not write dummy CSV: " + csv_path.string());
    }

    for (int feature = 0; feature < num_features; ++feature)
    {
        stream << "f" << feature << ",";
    }
    stream << "label\n";

    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0F, 0.15F);
    for (int row = 0; row < num_samples; ++row)
    {
        const int label = row % num_classes;
        for (int feature = 0; feature < num_features; ++feature)
        {
            const float value = (feature == label ? 1.0F : 0.0F) + noise(rng);
            stream << value << ",";
        }
        stream << label << "\n";
    }
}

inline auto one_hot_labels(const std::vector<float>& class_ids, int num_classes) -> std::vector<float>
{
    std::vector<float> encoded(class_ids.size() * static_cast<std::size_t>(num_classes), 0.0F);
    for (std::size_t row = 0; row < class_ids.size(); ++row)
    {
        int label = static_cast<int>(std::lround(class_ids[row]));
        label = std::clamp(label, 0, num_classes - 1);
        encoded[(row * static_cast<std::size_t>(num_classes)) + static_cast<std::size_t>(label)] = 1.0F;
    }
    return encoded;
}

inline auto argmax_row(const std::vector<float>& values, int row, int cols) -> int
{
    const std::size_t offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    int best = 0;
    float best_value = values[offset];
    for (int col = 1; col < cols; ++col)
    {
        const float value = values[offset + static_cast<std::size_t>(col)];
        if (value > best_value)
        {
            best_value = value;
            best = col;
        }
    }
    return best;
}
