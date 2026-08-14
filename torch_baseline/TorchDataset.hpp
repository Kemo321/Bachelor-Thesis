#pragma once

#include "DeepLearnLib/dataset.hpp"

#include <torch/torch.h>

/**
 * @brief LibTorch Dataset adapter for VOC-style YOLO data.
 *
 * Used only by --torch baseline benchmarks. The custom pipeline uses CustomDataLoader.
 */
class VOCYoloDataset : public torch::data::datasets::Dataset<VOCYoloDataset>
{
public:
    explicit VOCYoloDataset(const DataPaths& paths_param, bool is_train = false,
                            const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT);

    [[nodiscard]] auto get(size_t index) -> torch::data::Example<> override;
    [[nodiscard]] auto size() const -> torch::optional<size_t> override;

private:
    DataPaths paths;
    bool is_train_;
    const int img_size = 448;
    int num_classes_;
};
