#pragma once

#include "DeepLearnLib/Tensor.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

extern const std::vector<std::string> VOC_CLASSES_DEFAULT;

/**
 * @brief Container for dataset file paths.
 *
 * Holds parallel vectors of image file paths and corresponding label file paths.
 */
struct DataPaths
{
    std::vector<std::string> images;
    std::vector<std::string> labels;
};

/**
 * @brief Split a VOC-style dataset into train/validation/test sets.
 *
 * @param voc_root Root folder of the VOC dataset (e.g. contains JPEGImages/ and Annotations/).
 * @param[out] train DataPaths that will be populated with training image and label file paths.
 * @param[out] val DataPaths that will be populated with validation image and label file paths.
 * @param[out] test DataPaths that will be populated with test image and label file paths.
 * @param class_names Vector of class names to consider. Default is VOC_CLASSES_DEFAULT.
 * @param train_ratio Fraction of the dataset to use for training (default 0.7F).
 * @param val_ratio Fraction of the dataset to use for validation (default 0.15F).
 */
void split_dataset(const std::string& voc_root, DataPaths& train, DataPaths& val, DataPaths& test,
                   const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT, float train_ratio = 0.7F,
                   float val_ratio = 0.15F);

/**
 * @brief CamelCase compatibility wrapper for split_dataset.
 */
inline void splitDataset(const std::string& voc_root, DataPaths& train, DataPaths& val, DataPaths& test,
                         const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT, float train_ratio = 0.7F,
                         float val_ratio = 0.15F)
{
    split_dataset(voc_root, train, val, test, class_names, train_ratio, val_ratio);
}

/**
 * @brief One GPU-resident training/evaluation batch.
 *
 * images:  [Batch, 3, 448, 448] in CHW layout.
 * targets: [Batch, 7, 7, 10 + num_classes] YOLOv1 grid encoding.
 */
struct Batch
{
    dl::Tensor images;
    dl::Tensor targets;
};

/**
 * @brief Sequential/shuffled mini-batch loader built on OpenCV and dl::Tensor.
 *
 * Training mode applies scale/translation (cv::warpAffine) and HSV saturation/exposure jitter
 * on the CPU, then uploads CHW image buffers and YOLO targets with dl::Tensor::from_host.
 */
class CustomDataLoader
{
public:
    CustomDataLoader(const DataPaths& paths, int batch_size, bool is_train,
                     const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT);

    /**
     * @brief Rewind to the start of an epoch. Shuffles sample order when is_train is true.
     */
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
