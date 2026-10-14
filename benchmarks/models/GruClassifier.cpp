#include "GruClassifier.hpp"

#include "DeepLearnLib/layers/FullyConnected.hpp"
#include "DeepLearnLib/layers/Gru.hpp"

#include <stdexcept>

GruClassifier::GruClassifier(int input_size, int hidden_size, int num_classes)
    : input_size_(input_size)
    , hidden_size_(hidden_size)
    , num_classes_(num_classes)
{
    if (input_size_ <= 0 || hidden_size_ <= 0 || num_classes_ <= 0)
    {
        throw std::runtime_error("GruClassifier requires positive input, hidden, and class counts");
    }

    layers_.push_back(std::make_shared<Gru>(input_size_, hidden_size_));
    layers_.push_back(std::make_shared<FullyConnected>(hidden_size_, num_classes_, 0.0F));
}

auto GruClassifier::forward_logits(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
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

auto GruClassifier::get_all_layers() -> std::vector<std::shared_ptr<Layer>>
{
    return layers_;
}

auto GruClassifier::input_size() const -> int
{
    return input_size_;
}

auto GruClassifier::hidden_size() const -> int
{
    return hidden_size_;
}

auto GruClassifier::num_classes() const -> int
{
    return num_classes_;
}
