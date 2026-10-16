#pragma once

#include <filesystem>
#include <string>
#include <vector>

struct Ecg200Data
{
    int length = 0;
    std::vector<float> values;
    std::vector<int> labels;
    std::vector<std::string> class_names;

    [[nodiscard]] auto size() const -> int
    {
        return static_cast<int>(labels.size());
    }
};

[[nodiscard]] auto load_ecg200_file(const std::filesystem::path& path) -> Ecg200Data;

[[nodiscard]] auto find_ecg200_split(const std::filesystem::path& root, const std::string& which)
    -> std::filesystem::path;
