#include "DeepLearnLib/Network.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace
{

template <typename T>
auto write_pod(std::ostream& stream, const T& value, const std::string& path) -> void
{
    stream.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!stream)
    {
        throw std::runtime_error("Failed to write model data to '" + path + "'");
    }
}

template <typename T>
auto read_pod(std::istream& stream, T& value, const std::string& path) -> void
{
    stream.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!stream)
    {
        throw std::runtime_error("Failed to read model data from '" + path + "' (unexpected EOF or I/O error)");
    }
}

auto write_bytes(std::ostream& stream, const void* data, std::size_t bytes, const std::string& path) -> void
{
    if (bytes == 0)
    {
        return;
    }
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream)
    {
        throw std::runtime_error("Failed to write model payload to '" + path + "'");
    }
}

auto read_bytes(std::istream& stream, void* data, std::size_t bytes, const std::string& path) -> void
{
    if (bytes == 0)
    {
        return;
    }
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream)
    {
        throw std::runtime_error("Failed to read model payload from '" + path + "' (unexpected EOF or I/O error)");
    }
}

} // namespace

Network::Network(std::vector<std::shared_ptr<Layer>> layers_vector, float learning_rate_val, float gradient_clip)
    : layers_(std::move(layers_vector))
    , gradient_clip_(gradient_clip)
{
    for (const auto& layer_pointer : layers_)
    {
        if (layer_pointer == nullptr)
        {
            throw std::runtime_error("Network cannot contain a null layer");
        }
        layer_pointer->learning_rate = learning_rate_val;
        layer_pointer->gradient_clip = gradient_clip_;
    }
}

void Network::set_gradient_clip(float abs_bound)
{
    gradient_clip_ = abs_bound;
    sync_layer_optimizer_state();
}

auto Network::sync_layer_optimizer_state() -> void
{
    for (auto& layer : layers_)
    {
        layer->gradient_clip = gradient_clip_;
    }
}

auto Network::gradient_clip() const -> float
{
    return gradient_clip_;
}

auto Network::clip_loss_gradient(const dl::Tensor& gradient) const -> dl::Tensor
{
    if (gradient_clip_ <= 0.0F)
    {
        return gradient.as_view();
    }
    const float clip = dl::scaled_gradient_clip(gradient_clip_);
    dl::Tensor& clipped = dl::Tensor::ensure(loss_grad_clip_cache_, gradient.get_shape(), dl::Device::GPU,
        gradient.get_dtype());
    if (gradient.get_size() > 0)
    {
        dl::memcpy_d2d_on_current(clipped.data(), gradient.data(), gradient.nbytes());
        clipped.clamp_(-clip, clip);
    }
    return clipped.as_view();
}

void Network::clip_parameter_gradients(cudaStream_t stream)
{
    if (gradient_clip_ <= 0.0F)
    {
        return;
    }
    const float clip = dl::scaled_gradient_clip(gradient_clip_);
    for (auto& layer : layers_)
    {
        layer->clip_gradients(clip, stream);
    }
}

auto Network::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::StreamGuard stream_guard(stream);
    dl::Tensor current = input_tensor.view(input_tensor.get_shape());
    for (std::size_t index = 0; index < layers_.size(); ++index)
    {
        current = layers_[index]->forward(current, stream);
        current = current.view(current.get_shape());
#ifdef DEBUG_NUMERICS
        const std::string context = "Layer " + std::to_string(index) + " (" + typeid(*layers_[index]).name()
            + ") forward";
        current.assert_finite(context.c_str());
#else
        (void)index;
#endif
    }
    return current;
}

auto Network::fit(const dl::Tensor& x_train, const dl::Tensor& y_train, int epochs, int verbose) -> void
{
    if (epochs < 0)
    {
        throw std::runtime_error("Network::fit requires a non-negative epoch count");
    }

    for (int epoch_idx = 0; epoch_idx < epochs; ++epoch_idx)
    {
        dl::Tensor prediction = forward(x_train);

        constexpr int log_interval = 10;
        const bool should_log = verbose != 0 && (epoch_idx % log_interval == 0 || epoch_idx == epochs - 1);
        float loss_value = 0.0F;
        if (should_log)
        {
            const std::vector<float> loss_host = YOLOLoss::loss(y_train, prediction).to_host();
            if (loss_host.empty())
            {
                throw std::runtime_error("YOLOLoss::loss returned an empty tensor");
            }
            loss_value = loss_host.front();
        }

        dl::Tensor gradient_error = clip_loss_gradient(YOLOLoss::loss_derivative(y_train, prediction));

        std::size_t reverse_index = layers_.size();
        for (auto iterator = layers_.rbegin(); iterator != layers_.rend(); ++iterator)
        {
            --reverse_index;
            gradient_error = (*iterator)->backward(gradient_error);
#ifdef DEBUG_NUMERICS
            const std::string context = "Layer " + std::to_string(reverse_index) + " (" + typeid(**iterator).name()
                + ") backward";
            gradient_error.assert_finite(context.c_str());
#else
            (void)reverse_index;
#endif
        }

        for (auto& layer : layers_)
        {
            layer->step();
        }

        if (should_log)
        {
            dl::log_info_message("Epoch " + std::to_string(epoch_idx) + "/" + std::to_string(epochs)
                + " | Loss: " + std::to_string(loss_value));
        }
    }
}

auto Network::save(const std::string& path) -> void
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Failed to open '" + path + "' for writing");
    }

    const auto layer_count = static_cast<std::int32_t>(layers_.size());
    write_pod(stream, layer_count, path);

    for (const auto& layer : layers_)
    {
        auto parameters = layer->get_parameters();
        const auto param_count = static_cast<std::int32_t>(parameters.size());
        write_pod(stream, param_count, path);

        for (auto& parameter : parameters)
        {
            const auto name_length = static_cast<std::int32_t>(parameter.first.size());
            write_pod(stream, name_length, path);
            write_bytes(stream, parameter.first.data(), static_cast<std::size_t>(name_length), path);

            const std::vector<int>& shape = parameter.second.get_shape();
            const auto rank = static_cast<std::int32_t>(shape.size());
            write_pod(stream, rank, path);
            for (int dimension : shape)
            {
                write_pod(stream, static_cast<std::int32_t>(dimension), path);
            }

            const std::vector<float> host = parameter.second.to_host();
            write_bytes(stream, host.data(), host.size() * sizeof(float), path);
        }
    }

    stream.flush();
    if (!stream)
    {
        throw std::runtime_error("Failed to flush model file '" + path + "'");
    }
    dl::log_info_message("Model saved to: " + path);
}

auto Network::load(const std::string& path) -> void
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Failed to open '" + path + "' for reading");
    }

    std::int32_t layer_count { 0 };
    read_pod(stream, layer_count, path);
    if (layer_count != static_cast<std::int32_t>(layers_.size()))
    {
        throw std::runtime_error("Model file '" + path + "' layer count does not match the current network");
    }

    for (auto& layer : layers_)
    {
        std::int32_t param_count { 0 };
        read_pod(stream, param_count, path);

        std::map<std::string, dl::Tensor> loaded;
        for (std::int32_t param_idx = 0; param_idx < param_count; ++param_idx)
        {
            std::int32_t name_length { 0 };
            read_pod(stream, name_length, path);
            if (name_length < 0)
            {
                throw std::runtime_error("Invalid parameter name length in '" + path + "'");
            }

            std::string name(static_cast<std::size_t>(name_length), '\0');
            read_bytes(stream, name.data(), static_cast<std::size_t>(name_length), path);

            std::int32_t rank { 0 };
            read_pod(stream, rank, path);
            if (rank < 0)
            {
                throw std::runtime_error("Invalid tensor rank in '" + path + "'");
            }

            std::vector<int> shape(static_cast<std::size_t>(rank));
            std::size_t numel = 1;
            for (std::int32_t dim_idx = 0; dim_idx < rank; ++dim_idx)
            {
                std::int32_t dimension { 0 };
                read_pod(stream, dimension, path);
                shape[static_cast<std::size_t>(dim_idx)] = dimension;
                numel *= static_cast<std::size_t>(dimension);
            }
            if (rank == 0)
            {
                numel = 1;
            }

            std::vector<float> host(numel);
            read_bytes(stream, host.data(), host.size() * sizeof(float), path);
            loaded.emplace(std::move(name), dl::Tensor::from_host(shape, host, dl::Device::GPU));
        }

        layer->set_parameters(loaded);
    }

    dl::log_info_message("Model loaded from: " + path);
}
