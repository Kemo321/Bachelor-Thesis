#include "FusedCbrCNN.hpp"

#include "cnn_shapes.hpp"

#include "DeepLearnLib/layers/Flatten.hpp"
#include "DeepLearnLib/layers/FullyConnected.hpp"
#include "DeepLearnLib/layers/FusedCBR2d.hpp"
#include "DeepLearnLib/layers/MaxPool2d.hpp"

#include <stdexcept>

FusedCbrCNN::FusedCbrCNN(int num_classes, int image_size, int in_channels)
    : num_classes_(num_classes)
    , image_size_(image_size)
    , in_channels_(in_channels)
{
    if (num_classes_ <= 0 || image_size_ <= 0 || in_channels_ <= 0)
    {
        throw std::runtime_error("FusedCbrCNN requires positive class, image, and channel counts");
    }

    layers_.push_back(std::make_shared<FusedCBR2d>(in_channels_, 16, 3, 1, 1, 0.1F, 1.0e-5F, 0.1F));
    layers_.push_back(std::make_shared<MaxPool2d>(2, 2));
    layers_.push_back(std::make_shared<FusedCBR2d>(16, 32, 3, 1, 1, 0.1F, 1.0e-5F, 0.1F));
    layers_.push_back(std::make_shared<MaxPool2d>(2, 2));
    layers_.push_back(std::make_shared<Flatten>());
    layers_.push_back(std::make_shared<FullyConnected>(pooled_cnn_features(image_size_), num_classes_, 0.0F));
}

auto FusedCbrCNN::forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    dl::Tensor current = input_tensor.view(input_tensor.get_shape());
    for (auto& layer : layers_)
    {
        current = layer->forward(current, stream);
        current = current.view(current.get_shape());
    }
    return current;
}

auto FusedCbrCNN::get_all_layers() -> std::vector<std::shared_ptr<Layer>>
{
    return layers_;
}

auto FusedCbrCNN::num_classes() const -> int
{
    return num_classes_;
}

auto FusedCbrCNN::image_size() const -> int
{
    return image_size_;
}

auto FusedCbrCNN::in_channels() const -> int
{
    return in_channels_;
}
