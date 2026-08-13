#pragma once

#include <string>
#include <torch/torch.h>
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
void split_dataset(const std::string& voc_root, DataPaths& train, DataPaths& val, DataPaths& test, const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT, float train_ratio = 0.7F, float val_ratio = 0.15F);

/**
 * @brief CamelCase compatibility wrapper for split_dataset.
 *
 * This wrapper preserves the original split_dataset function while providing a
 * camelCase API. It simply forwards all arguments to split_dataset.
 */
inline void splitDataset(const std::string& voc_root, DataPaths& train, DataPaths& val, DataPaths& test, const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT, float train_ratio = 0.7F, float val_ratio = 0.15F)
{
    split_dataset(voc_root, train, val, test, class_names, train_ratio, val_ratio);
}

/**
 * @brief Dataset adapter for VOC formatted data tailored for YOLO-style models.
 *
 * This class implements the Dataset concept from LibTorch and provides
 * tensors suitable for training and validation of YOLO-like networks.
 */
class VOCYoloDataset : public torch::data::datasets::Dataset<VOCYoloDataset>
{
public:
    /**
     * @brief Construct a new VOCYoloDataset object.
     *
     * @param paths_param DataPaths containing image and label file paths.
     * @param is_train Whether the dataset should operate in training mode (applies augmentations).
     * @param class_names Names of classes to consider; used to map labels to indices.
     */
    explicit VOCYoloDataset(const DataPaths& paths_param, bool is_train = false, const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT);

    /**
     * @brief Retrieve the dataset example at the given index.
     *
     * @param index Index of the example to retrieve.
     * @return torch::data::Example<> Pair of tensors {input, target}.
     *         - input: [Channels, Height, Width] (typical image tensor layout).
     *         - target: shape depends on label encoding used by the project (see implementation).
     */
    [[nodiscard]] auto get(size_t index) -> torch::data::Example<> override;

    /**
     * @brief Return the size (number of samples) in the dataset.
     *
     * @return torch::optional<size_t> Number of samples.
     */
    [[nodiscard]] auto size() const -> torch::optional<size_t> override;

private:
    DataPaths paths;
    bool is_train_;
    const int img_size = 448;
    int num_classes_;
};