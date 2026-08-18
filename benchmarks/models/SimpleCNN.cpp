#include "SimpleCNN.hpp"

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Flatten.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"

#include <stdexcept>

namespace
{

auto conv_out(int size, int kernel, int stride, int padding) -> int
{
    return ((size + (2 * padding) - kernel) / stride) + 1;
}

auto flatten_features(int image_size) -> int
{
    int spatial = conv_out(image_size, 3, 1, 1);
    spatial = conv_out(spatial, 2, 2, 0);
    spatial = conv_out(spatial, 3, 1, 1);
    spatial = conv_out(spatial, 2, 2, 0);
    return 32 * spatial * spatial;
}

} // namespace

SimpleCNN::SimpleCNN(int num_classes, int image_size, int in_channels)
    : num_classes_(num_classes)
    , image_size_(image_size)
    , in_channels_(in_channels)
    , softmax_(std::make_shared<Softmax>())
{
    if (num_classes_ <= 0)
    {
        throw std::runtime_error("SimpleCNN requires a positive class count");
    }
    if (image_size_ <= 0)
    {
        throw std::runtime_error("SimpleCNN requires a positive image size");
    }
    if (in_channels_ <= 0)
    {
        throw std::runtime_error("SimpleCNN requires a positive input channel count");
    }

    layers_.push_back(std::make_shared<Conv2d>(in_channels_, 16, 3, 1, 1));
    layers_.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers_.push_back(std::make_shared<MaxPool2d>(2, 2));
    layers_.push_back(std::make_shared<Conv2d>(16, 32, 3, 1, 1));
    layers_.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers_.push_back(std::make_shared<MaxPool2d>(2, 2));
    layers_.push_back(std::make_shared<Flatten>());
    layers_.push_back(std::make_shared<FullyConnected>(flatten_features(image_size_), num_classes_));
    LOG_DEBUG("SimpleCNN classes={} image_size={} in_channels={} flatten_features={} layers={}", num_classes_,
        image_size_, in_channels_, flatten_features(image_size_), layers_.size());
}

auto SimpleCNN::forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
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

auto SimpleCNN::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    return softmax_->forward(forward_logits(input_tensor, stream), stream);
}

auto SimpleCNN::get_all_layers() -> std::vector<std::shared_ptr<Layer>>
{
    return layers_;
}

auto SimpleCNN::num_classes() const -> int
{
    return num_classes_;
}

auto SimpleCNN::image_size() const -> int
{
    return image_size_;
}

auto SimpleCNN::in_channels() const -> int
{
    return in_channels_;
}
