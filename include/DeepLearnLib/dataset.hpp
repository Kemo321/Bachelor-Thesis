
#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>

struct DataPaths
{
    std::vector<std::string> images;
    std::vector<std::string> labels;
};

void split_dataset(const std::string& voc_root,
                   DataPaths& train, DataPaths& val, DataPaths& test,
                   float train_ratio = 0.7F, float val_ratio = 0.15F);

class VOCYoloDataset : public torch::data::datasets::Dataset<VOCYoloDataset>
{
private:
    DataPaths paths;
    const int img_size = 448;

public:
    explicit VOCYoloDataset(const DataPaths& p);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};