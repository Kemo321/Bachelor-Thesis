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

SimpleCNN::SimpleCNN(int num_classes, int image_size)
    : num_classes_(num_classes)
    , image_size_(image_size)
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

    layers_.push_back(std::make_shared<Conv2d>(3, 16, 3, 1, 1));
    layers_.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers_.push_back(std::make_shared<MaxPool2d>(2, 2));
    layers_.push_back(std::make_shared<Conv2d>(16, 32, 3, 1, 1));
    layers_.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers_.push_back(std::make_shared<MaxPool2d>(2, 2));
    layers_.push_back(std::make_shared<Flatten>());
    layers_.push_back(std::make_shared<FullyConnected>(flatten_features(image_size_), num_classes_));
    LOG_INFO("SimpleCNN classes={} image_size={} flatten_features={} layers={}", num_classes_, image_size_,
        flatten_features(image_size_), layers_.size());
    LOG_FLUSH();
}

auto SimpleCNN::forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    dl::Tensor current = input_tensor.view(input_tensor.get_shape());
    static thread_local bool logged_shapes = false;
    const bool dump_shapes = !logged_shapes;
    if (dump_shapes)
    {
        LOG_INFO("SimpleCNN first forward input {}", input_tensor.describe());
    }
    for (std::size_t index = 0; index < layers_.size(); ++index)
    {
        current = layers_[index]->forward(current, stream);
        current = current.view(current.get_shape());
        if (dump_shapes)
        {
            LOG_INFO("SimpleCNN layer {}/{} -> {}", index + 1, layers_.size(), current.describe());
        }
    }
    if (dump_shapes)
    {
        logged_shapes = true;
        LOG_FLUSH();
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
