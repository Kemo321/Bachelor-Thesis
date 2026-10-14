#include "DeepLearnLib/core/Network.hpp"

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

// Writes a POD value. The path is used only in the message when the stream reports an error.
template <typename T>
auto write_pod(std::ostream& stream, const T& value, const std::string& path) -> void
{
    stream.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!stream)
    {
        throw std::runtime_error("Failed to write model data to '" + path + "'");
    }
}

// Reads a POD value. A short read is treated as a truncated weight file.
template <typename T>
auto read_pod(std::istream& stream, T& value, const std::string& path) -> void
{
    stream.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!stream)
    {
        throw std::runtime_error("Failed to read model data from '" + path + "' (unexpected EOF or I/O error)");
    }
}

// Writes a raw buffer. An empty payload is skipped so write is not called with length zero.
auto write_bytes(std::ostream& stream, const void* data, std::size_t bytes, const std::string& path) -> void
{
    if (bytes == 0)
    {
        // Skip a zero-length write; the stream is not touched.
        return;
    }
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream)
    {
        throw std::runtime_error("Failed to write model payload to '" + path + "'");
    }
}

// Reads a raw buffer. An empty payload is skipped.
auto read_bytes(std::istream& stream, void* data, std::size_t bytes, const std::string& path) -> void
{
    if (bytes == 0)
    {
        // Skip a zero-length read.
        return;
    }
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream)
    {
        throw std::runtime_error("Failed to read model payload from '" + path + "' (unexpected EOF or I/O error)");
    }
}

} // namespace

// Takes ownership of the layers and writes them a shared learning rate and gradient-clip threshold.
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

// Changes the clip threshold and copies it into the layers immediately.
void Network::set_gradient_clip(float abs_bound)
{
    gradient_clip_ = abs_bound;
    sync_layer_optimizer_state();
}

// Copies gradient_clip_ into the gradient_clip field of every layer.
auto Network::sync_layer_optimizer_state() -> void
{
    for (auto& layer : layers_)
    {
        layer->gradient_clip = gradient_clip_;
    }
}

// The network's current gradient-clip threshold.
auto Network::gradient_clip() const -> float
{
    return gradient_clip_;
}

// Clips the loss gradient to +/- gradient_clip_. A threshold <= 0 returns a view and does not copy.
auto Network::clip_loss_gradient(const dl::Tensor& gradient) const -> dl::Tensor
{
    if (gradient_clip_ <= 0.0F)
    {
        // A non-positive threshold leaves the gradient as a view, with no copy and no clamp.
        return gradient.as_view();
    }
    const float clip = gradient_clip_;
    // ensure reuses this slot, so an unchanged shape does not cudaMalloc every step.
    dl::Tensor& clipped = dl::Tensor::ensure(loss_grad_clip_cache_, gradient.get_shape(), dl::Device::GPU,
        gradient.get_dtype());
    if (gradient.get_size() > 0)
    {
        dl::memcpy_d2d_on_current(clipped.data(), gradient.data(), gradient.nbytes());
        clipped.clamp_(-clip, clip);
    }
    return clipped.as_view();
}

// Calls clip_gradients on every layer. A threshold <= 0 does nothing.
void Network::clip_parameter_gradients(cudaStream_t stream)
{
    if (gradient_clip_ <= 0.0F)
    {
        return;
    }
    const float clip = gradient_clip_;
    for (auto& layer : layers_)
    {
        layer->clip_gradients(clip, stream);
    }
}

// Runs the input through the layers in order on the given stream.
// view after each layer keeps the shape without copying the data.
auto Network::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    // The guard binds this stream on the thread and on cuBLAS for the whole forward.
    const dl::StreamGuard stream_guard(stream);
    dl::Tensor current = input_tensor.view(input_tensor.get_shape());
    for (std::size_t index = 0; index < layers_.size(); ++index)
    {
        // view keeps this activation's shape on the same buffer; it does not copy the data.
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

// Writes the layer count and, for every parameter, the name, the shape, and the fp32 weights.
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

            // to_host synchronizes so these fp32 weights are a host buffer before they are written.
            const std::vector<float> host = parameter.second.to_host();
            write_bytes(stream, host.data(), host.size() * sizeof(float), path);
        }
    }

    stream.flush();
    if (!stream)
    {
        throw std::runtime_error("Failed to flush model file '" + path + "'");
    }
}

// Reads that same layout and calls set_parameters. The layer count must match the network.
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
                // A scalar has an empty shape, so the product loop never ran; the count is 1.
                numel = 1;
            }

            std::vector<float> host(numel);
            read_bytes(stream, host.data(), host.size() * sizeof(float), path);
            loaded.emplace(std::move(name), dl::Tensor::from_host(shape, host, dl::Device::GPU));
        }

        layer->set_parameters(loaded);
    }
}
