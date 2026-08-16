#include "DeepLearnLib/YOLO.hpp"

#include "DeepLearnLib/Dropout.hpp"
#include "DeepLearnLib/Flatten.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/FusedCBR2d.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"

YOLO::YOLO(int num_classes)
{
    auto add_block = [&](int in_channels, int out_channels, int kernel, int stride, int padding)
    {
        backbone_layers.push_back(
            std::make_shared<FusedCBR2d>(in_channels, out_channels, kernel, stride, padding, 0.1F));
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

    for (int block_idx = 0; block_idx < 4; ++block_idx)
    {
        add_block(512, 256, 1, 1, 0);
        add_block(256, 512, 3, 1, 1);
    }
    add_block(512, 512, 1, 1, 0);
    add_block(512, 1024, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    for (int block_idx = 0; block_idx < 2; ++block_idx)
    {
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
    head_layers.push_back(std::make_shared<FullyConnected>(4096, 7 * 7 * (10 + num_classes), 0.9F));
}

auto YOLO::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    dl::Tensor current = input_tensor.view(input_tensor.get_shape());
    for (auto& layer : backbone_layers)
    {
        current = layer->forward(current, stream);
        current = current.view(current.get_shape());
    }
    for (auto& layer : head_layers)
    {
        current = layer->forward(current, stream);
        current = current.view(current.get_shape());
    }
    return current;
}

auto YOLO::get_all_layers() -> std::vector<std::shared_ptr<Layer>>
{
    std::vector<std::shared_ptr<Layer>> all_layers;
    all_layers.reserve(backbone_layers.size() + head_layers.size());
    all_layers.insert(all_layers.end(), backbone_layers.begin(), backbone_layers.end());
    all_layers.insert(all_layers.end(), head_layers.begin(), head_layers.end());
    return all_layers;
}
