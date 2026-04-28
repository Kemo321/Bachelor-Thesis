#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Flatten.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Dropout.hpp"


struct YOLOImpl : torch::nn::Module
{
    std::vector<std::shared_ptr<Layer>> backbone_layers;
    std::vector<std::shared_ptr<Layer>> head_layers;

    YOLOImpl();

    [[nodiscard]] auto forward(torch::Tensor input_tensor) -> torch::Tensor;
    [[nodiscard]] auto get_all_layers() -> std::vector<std::shared_ptr<Layer>>;
};

TORCH_MODULE(YOLO);

[[nodiscard]] auto compute_manual_yolo_loss_gradient(const torch::Tensor& prediction, const torch::Tensor& target) -> torch::Tensor;