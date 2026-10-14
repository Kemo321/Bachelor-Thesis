#pragma once

#include <filesystem>
#include <string>
#include <vector>

/**
 * One split of the UCR ECG200 archive.
 *
 * Each row is one heartbeat of `length` samples and a class id. File labels are
 * sorted numerically, so `-1` becomes class 0 and `1` becomes class 1.
 */
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

/** Read a UCR `.txt` / `.tsv` row or an aeon `.ts` row. */
[[nodiscard]] auto load_ecg200_file(const std::filesystem::path& path) -> Ecg200Data;

/**
 * Find `ECG200_<which>.txt`, then `.tsv`, then `.ts` under @p root.
 * @p which is `TRAIN` or `TEST`.
 */
[[nodiscard]] auto find_ecg200_split(const std::filesystem::path& root, const std::string& which)
    -> std::filesystem::path;
