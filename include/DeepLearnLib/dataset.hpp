#pragma once

#include "DeepLearnLib/Tensor.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

extern const std::vector<std::string> VOC_CLASSES_DEFAULT;

/**
 * Parallel lists of image paths and matching annotation paths.
 */
struct DataPaths
{
    std::vector<std::string> images;
    std::vector<std::string> labels;
};

void split_dataset(const std::string& voc_root, DataPaths& train, DataPaths& val, DataPaths& test,
    const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT, float train_ratio = 0.7F,
    float val_ratio = 0.15F);

inline void splitDataset(const std::string& voc_root, DataPaths& train, DataPaths& val, DataPaths& test,
    const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT, float train_ratio = 0.7F,
    float val_ratio = 0.15F)
{
    split_dataset(voc_root, train, val, test, class_names, train_ratio, val_ratio);
}

/**
 * One GPU-resident training/evaluation batch.
 *
 * images:  [Batch, 3, 448, 448] CHW.
 * targets: [Batch, 7, 7, 10 + num_classes] YOLOv1 grid.
 */
struct Batch
{
    dl::Tensor images;
    dl::Tensor targets;
};

/**
 * Sequential/shuffled mini-batch loader (OpenCV decode + dl::Tensor upload).
 *
 * Training applies affine scale/translation and HSV jitter on the CPU, then
 * uploads CHW images and YOLO targets with from_host.
 */
class CustomDataLoader
{
public:
    CustomDataLoader(const DataPaths& paths, int batch_size, bool is_train,
        const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT);

    auto reset() -> void;
    [[nodiscard]] auto has_next() const -> bool;
    auto get_batch() -> Batch;
    [[nodiscard]] auto size() const -> std::size_t;
    [[nodiscard]] auto batch_size() const -> int;

private:
    DataPaths paths_;
    int batch_size_;
    bool is_train_;
    int num_classes_;
    int img_size_;
    std::size_t cursor_;
    std::vector<std::size_t> order_;
    std::mt19937 rng_;

    auto load_sample(std::size_t sample_index, std::vector<float>& image_chw, std::vector<float>& target) -> void;
};
