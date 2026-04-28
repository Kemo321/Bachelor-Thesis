#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/BatchNorm2d.hpp"

YOLOImpl::YOLOImpl() {
    auto add_block = [&](int in, int out, int k, int s, int p) {
        backbone_layers.push_back(std::make_shared<Conv2d>(in, out, k, s, p));
        backbone_layers.push_back(std::make_shared<BatchNorm2d>(out));
        backbone_layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    };

    add_block(3, 64, 7, 2, 3);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    add_block(64, 192, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    add_block(192, 128, 1, 1, 0);
    add_block(128, 256, 3, 1, 1);
    add_block(256, 256, 1, 1, 0);
    add_block(256, 512, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    for (int i = 0; i < 4; ++i) {
        add_block(512, 256, 1, 1, 0);
        add_block(256, 512, 3, 1, 1);
    }
    add_block(512, 512, 1, 1, 0);
    add_block(512, 1024, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    for (int i = 0; i < 2; ++i) {
        add_block(1024, 512, 1, 1, 0);
        add_block(512, 1024, 3, 1, 1);
    }
    add_block(1024, 1024, 3, 1, 1);
    add_block(1024, 1024, 3, 2, 1);
    add_block(1024, 1024, 3, 1, 1);
    add_block(1024, 1024, 3, 1, 1);

    head_layers.push_back(std::make_shared<Flatten>());
    head_layers.push_back(std::make_shared<FullyConnected>(7 * 7 * 1024, 4096, 0.9F));
    head_layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    head_layers.push_back(std::make_shared<Dropout>(0.5F));
    head_layers.push_back(std::make_shared<FullyConnected>(4096, 7 * 7 * 30, 0.9F));
}

auto YOLOImpl::forward(torch::Tensor x) -> torch::Tensor {
    for (auto& layer : backbone_layers) {
        x = layer->forward(x);
    }
    for (auto& layer : head_layers) {
        x = layer->forward(x);
    }
    return x;
}

auto YOLOImpl::get_all_layers() -> std::vector<std::shared_ptr<Layer>>
{
    std::vector<std::shared_ptr<Layer>> all_layers;
    all_layers.reserve(backbone_layers.size() + head_layers.size());
    all_layers.insert(all_layers.end(), backbone_layers.begin(), backbone_layers.end());
    all_layers.insert(all_layers.end(), head_layers.begin(), head_layers.end());
    return all_layers;
}