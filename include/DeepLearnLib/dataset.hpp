#pragma once

#include <string>
#include <torch/torch.h>
#include <vector>

extern const std::vector<std::string> VOC_CLASSES_DEFAULT;

struct DataPaths
{
    std::vector<std::string> images;
    std::vector<std::string> labels;
};

void split_dataset(const std::string& voc_root, DataPaths& train, DataPaths& val, DataPaths& test, const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT, float train_ratio = 0.7F, float val_ratio = 0.15F);

class VOCYoloDataset : public torch::data::datasets::Dataset<VOCYoloDataset>
{
public:
    explicit VOCYoloDataset(const DataPaths& paths_param, bool is_train = false, const std::vector<std::string>& class_names = VOC_CLASSES_DEFAULT);
    
    [[nodiscard]] auto get(size_t index) -> torch::data::Example<> override;
    [[nodiscard]] auto size() const -> torch::optional<size_t> override;

private:
    DataPaths paths;
    bool is_train_;
    const int img_size = 448;
    int num_classes_; 
};