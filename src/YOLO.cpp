#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/BatchNorm2d.hpp"

/**
 * @brief Construct a YOLO model implementation.
 *
 * Builds the backbone and head layers following the original YOLO architecture.
 * BatchNorm and LeakyReLU are applied after convolutions to stabilize training
 * and improve gradient flow. MaxPool down-samples spatial dimensions.
 *
 * @param num_classes Number of object classes. The final fully connected layer
 *        produces a tensor of shape [Batch, 7*7*(10 + num_classes)].
 */
YOLOImpl::YOLOImpl(int num_classes) {
    // Helper to add Conv -> BatchNorm -> LeakyReLU blocks to the backbone.
    // Naming: use camelCase for local functions.
    auto addBlock = [&](int in_channels, int out_channels, int kernel, int stride, int padding) {
        backbone_layers.push_back(std::make_shared<Conv2d>(in_channels, out_channels, kernel, stride, padding));
        backbone_layers.push_back(std::make_shared<BatchNorm2d>(out_channels));
        backbone_layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    };

    addBlock(3, 64, 7, 2, 3);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    addBlock(64, 192, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    addBlock(192, 128, 1, 1, 0);
    addBlock(128, 256, 3, 1, 1);
    addBlock(256, 256, 1, 1, 0);
    addBlock(256, 512, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    for (int i = 0; i < 4; ++i) {
        addBlock(512, 256, 1, 1, 0);
        addBlock(256, 512, 3, 1, 1);
    }
    addBlock(512, 512, 1, 1, 0);
    addBlock(512, 1024, 3, 1, 1);
    backbone_layers.push_back(std::make_shared<MaxPool2d>(2, 2));

    for (int i = 0; i < 2; ++i) {
        addBlock(1024, 512, 1, 1, 0);
        addBlock(512, 1024, 3, 1, 1);
    }
    addBlock(1024, 1024, 3, 1, 1);
    addBlock(1024, 1024, 3, 2, 1);
    addBlock(1024, 1024, 3, 1, 1);
    addBlock(1024, 1024, 3, 1, 1);

    head_layers.push_back(std::make_shared<Flatten>());
    head_layers.push_back(std::make_shared<FullyConnected>(7 * 7 * 1024, 4096, 0.9F));
    head_layers.push_back(std::make_shared<LeakyReLU>(0.1F));
    head_layers.push_back(std::make_shared<Dropout>(0.5F));
    head_layers.push_back(std::make_shared<FullyConnected>(4096, 7 * 7 * (10 + num_classes), 0.9F));
}

/**
 * @brief Forward pass through the YOLO network.
 *
 * Iteratively applies backbone layers to extract hierarchical features and
 * then applies the head to produce final detections. Each intermediate
 * tensor is made contiguous to ensure memory layout consistency on GPU and
 * to avoid potential errors when later views/permutations are applied.
 *
 * @param x Input tensor with shape [Batch, Channels, H, W].
 * @return torch::Tensor Output tensor with shape [Batch, 7*7*(10 + num_classes)].
 */
auto YOLOImpl::forward(torch::Tensor x) -> torch::Tensor {
    for (auto& layer : backbone_layers) {
        x = layer->forward(x);
        // Ensure contiguous memory after each operation to avoid fragmented tensors
        // which can cause issues with subsequent view/permute operations on CUDA.
        x = x.contiguous();
    }
    for (auto& layer : head_layers) {
        x = layer->forward(x);
        x = x.contiguous();
    }
    return x;
}

/**
 * @brief Retrieve all layers (backbone + head) in a single vector.
 *
 * Useful for iterating over every layer for operations such as weight
 * initialization, exporting, or parameter inspection.
 *
 * @return std::vector<std::shared_ptr<Layer>> Vector containing shared_ptrs to
 *         all layers in the model. The order is backbone layers first, then head layers.
 */
auto YOLOImpl::get_all_layers() -> std::vector<std::shared_ptr<Layer>>
{
    std::vector<std::shared_ptr<Layer>> all_layers;
    all_layers.reserve(backbone_layers.size() + head_layers.size());
    all_layers.insert(all_layers.end(), backbone_layers.begin(), backbone_layers.end());
    all_layers.insert(all_layers.end(), head_layers.begin(), head_layers.end());
    return all_layers;
}