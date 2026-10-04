#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Network.hpp"

// Short layer list for Network::forward: two Conv–LeakyReLU–MaxPool blocks and two FC layers.
// The first FC declares an input of 7*7*1024; after the second pooling this list does not add the convolutions that produce that size.
// This file has no Flatten layer. The network learning rate is 0.001.
auto create_yolo_v1() -> Network
{
    std::vector<std::shared_ptr<Layer>> layers;

    // Input block: 7×7 convolution, stride 2, padding 3 (3→64), LeakyReLU(0.1), 2×2 pooling.
    layers.push_back(std::make_shared<Conv2d>(3, 64, 7, 2, 3));
    layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    // Second block: 3×3 convolution, padding 1 (64→192), LeakyReLU(0.1), 2×2 pooling.
    layers.push_back(std::make_shared<Conv2d>(64, 192, 3, 1, 1));
    layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    // Linear head: 7*7*1024 → 4096 → 7*7*30, with LeakyReLU(0.1) between the layers.
    layers.push_back(std::make_shared<FullyConnected>(7 * 7 * 1024, 4096));
    layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers.push_back(std::make_shared<FullyConnected>(4096, 7 * 7 * 30));

    return Network(std::move(layers), 0.001F);
}
