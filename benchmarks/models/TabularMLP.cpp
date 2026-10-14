#include "TabularMLP.hpp"

#include "DeepLearnLib/layers/FullyConnected.hpp"
#include "DeepLearnLib/layers/LeakyReLU.hpp"

#include <stdexcept>

TabularMLP::TabularMLP(int in_features, int hidden_size, int num_classes)
    : in_features_(in_features)
    , hidden_size_(hidden_size)
    , num_classes_(num_classes)
{
    if (in_features_ <= 0 || hidden_size_ <= 0 || num_classes_ <= 0)
    {
        throw std::runtime_error("TabularMLP requires positive feature, hidden, and class counts");
    }

    layers_.push_back(std::make_shared<FullyConnected>(in_features_, hidden_size_, 0.0F));
    layers_.push_back(std::make_shared<LeakyReLU>(0.1F));
    layers_.push_back(std::make_shared<FullyConnected>(hidden_size_, num_classes_, 0.0F));
}

auto TabularMLP::forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::StreamGuard stream_guard(stream);
    dl::Tensor current = input_tensor.view(input_tensor.get_shape());
    for (auto& layer : layers_)
    {
        current = layer->forward(current, stream);
        current = current.view(current.get_shape());
    }
    return current;
}

auto TabularMLP::get_all_layers() -> std::vector<std::shared_ptr<Layer>>
{
    return layers_;
}

auto TabularMLP::in_features() const -> int
{
    return in_features_;
}

auto TabularMLP::hidden_size() const -> int
{
    return hidden_size_;
}

auto TabularMLP::num_classes() const -> int
{
    return num_classes_;
}
