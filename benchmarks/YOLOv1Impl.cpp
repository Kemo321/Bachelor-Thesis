#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Network.hpp"

auto create_yolo_v1() -> Network
{
    std::vector<std::shared_ptr<Layer>> layers;

    // --- Backbone (Przykład pierwszych warstw) ---
    layers.push_back(std::make_shared<Conv2d>(3, 64, 7, 2, 3));
    layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    layers.push_back(std::make_shared<Conv2d>(64, 192, 3, 1, 1));
    layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    // --- Head ---
    // Flattening w Twoim Network::forward załatwi std::move/view
    layers.push_back(std::make_shared<FullyConnected>(7 * 7 * 1024, 4096, 0.9F));
    layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers.push_back(std::make_shared<FullyConnected>(4096, 7 * 7 * 30, 0.9F));

    return Network(std::move(layers), 0.001F);
}
