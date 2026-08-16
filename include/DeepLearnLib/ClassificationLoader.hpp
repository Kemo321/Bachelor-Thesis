#pragma once

#include "DeepLearnLib/dataset.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

/**
 * Folder-based image classification loader (root/<split>/<class_name>/*.jpg).
 *
 * Yields Batch tensors: images NCHW float in [0, 1], targets one-hot [N, C]
 * for CrossEntropyLoss.
 */
class ClassificationLoader
{
public:
    ClassificationLoader(std::string dataset_root, std::string split, int batch_size, int image_size = 32,
        bool shuffle = true);

    auto reset() -> void;
    [[nodiscard]] auto has_next() const -> bool;
    auto get_batch() -> Batch;
    [[nodiscard]] auto size() const -> std::size_t;
    [[nodiscard]] auto batch_size() const -> int;
    [[nodiscard]] auto num_classes() const -> int;
    [[nodiscard]] auto image_size() const -> int;
    [[nodiscard]] auto class_names() const -> const std::vector<std::string>&;

private:
    auto load_sample(std::size_t sample_index, std::vector<float>& image_chw) -> int;

    std::string split_root_;
    int batch_size_;
    int image_size_;
    bool shuffle_;
    std::vector<std::string> class_names_;
    std::vector<std::pair<std::string, int>> samples_;
    std::vector<std::size_t> order_;
    std::size_t cursor_ { 0 };
    std::mt19937 rng_;
};
