#include "YOLO.hpp"

#include "DeepLearnLib/Dropout.hpp"
#include "DeepLearnLib/Flatten.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/FusedCBR2d.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"

YOLO::YOLO(int num_classes)
{
    // One backbone block is convolution, batch-norm, and LeakyReLU(0.1) fused into FusedCBR2d.
    auto add_block = [&](int in_channels, int out_channels, int kernel, int stride, int padding)
    {
        backbone_layers.push_back(
            std::make_shared<FusedCBR2d>(in_channels, out_channels, kernel, stride, padding, 0.1F));
    };

    // Input 3×448×448: a 7×7 convolution with stride 2 yields 64×224, and pooling goes down to 112.
    add_block(3, 64, 7, 2, 3);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    // 64→192 on 112×112, pooling down to 56.
    add_block(64, 192, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    // 1×1 reduction and expansion to 512 channels on 56×56, pooling down to 28.
    add_block(192, 128, 1, 1, 0);
    add_block(128, 256, 3, 1, 1);
    add_block(256, 256, 1, 1, 0);
    add_block(256, 512, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    // Four repeats of 512→256→512, then 512→1024 and pooling down to 14.
    for (int block_idx = 0; block_idx < 4; ++block_idx)
    {
        add_block(512, 256, 1, 1, 0);
        add_block(256, 512, 3, 1, 1);
    }
    add_block(512, 512, 1, 1, 0);
    add_block(512, 1024, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    // Two repeats of 1024→512→1024, then 3×3 convolutions; the middle one has stride 2 and goes from 14 down to 7.
    for (int block_idx = 0; block_idx < 2; ++block_idx)
    {
        add_block(1024, 512, 1, 1, 0);
        add_block(512, 1024, 3, 1, 1);
    }
    add_block(1024, 1024, 3, 1, 1);
    add_block(1024, 1024, 3, 2, 1);
    add_block(1024, 1024, 3, 1, 1);
    add_block(1024, 1024, 3, 1, 1);

    // Head: flatten 7×7×1024, a 4096 layer, LeakyReLU, dropout 0.5, and an output of 7×7×(10 + class count).
    head_layers.push_back(std::make_shared<Flatten>());
    // inertia stays 0: in FullyConnected that is the GEMM beta when writing dW. SGD momentum lives in Layer::momentum.
    head_layers.push_back(std::make_shared<FullyConnected>(7 * 7 * 1024, 4096));
    head_layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    head_layers.push_back(std::make_shared<Dropout>(0.5F));
    head_layers.push_back(std::make_shared<FullyConnected>(4096, 7 * 7 * (10 + num_classes)));
}

// Backbone first, then the head. view keeps the shape after every layer.
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

// Trainable order: the whole backbone, then the head. The training loop walks this list backward.
auto YOLO::get_all_layers() -> std::vector<std::shared_ptr<Layer>>
{
    std::vector<std::shared_ptr<Layer>> all_layers;
    all_layers.reserve(backbone_layers.size() + head_layers.size());
    all_layers.insert(all_layers.end(), backbone_layers.begin(), backbone_layers.end());
    all_layers.insert(all_layers.end(), head_layers.begin(), head_layers.end());
    return all_layers;
}
