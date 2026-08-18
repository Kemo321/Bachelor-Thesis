#include "DeepLearnLib/DarknetWeights.hpp"
#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/FusedCBR2d.hpp"
#include "DeepLearnLib/LocalLayer.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

auto read_exact(std::ifstream& stream, void* dst, std::size_t bytes, const std::string& path) -> void
{
    if (bytes == 0)
    {
        return;
    }
    stream.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!stream)
    {
        throw std::runtime_error("Unexpected EOF while reading Darknet weights '" + path + "'");
    }
}

auto read_f32(std::ifstream& stream, std::size_t count, const std::string& path) -> std::vector<float>
{
    std::vector<float> values(count);
    read_exact(stream, values.data(), count * sizeof(float), path);
    return values;
}

auto remaining_bytes(std::ifstream& stream) -> std::size_t
{
    const auto pos = stream.tellg();
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    stream.seekg(pos);
    if (pos < 0 || end < pos)
    {
        return 0;
    }
    return static_cast<std::size_t>(end - pos);
}

auto upload(const std::vector<int>& shape, const std::vector<float>& host) -> dl::Tensor
{
    return dl::Tensor::from_host(shape, host, dl::Device::GPU);
}

auto zeros_like_bias(int out_channels) -> dl::Tensor
{
    return upload({ 1, out_channels, 1, 1 }, std::vector<float>(static_cast<std::size_t>(out_channels), 0.0F));
}

auto transpose_connected(const std::vector<float>& darknet, int inputs, int outputs) -> std::vector<float>
{
    std::vector<float> ours(static_cast<std::size_t>(inputs) * static_cast<std::size_t>(outputs));
    for (int out_idx = 0; out_idx < outputs; ++out_idx)
    {
        for (int in_idx = 0; in_idx < inputs; ++in_idx)
        {
            ours[(static_cast<std::size_t>(in_idx) * static_cast<std::size_t>(outputs)) + static_cast<std::size_t>(out_idx)] =
                darknet[(static_cast<std::size_t>(out_idx) * static_cast<std::size_t>(inputs)) + static_cast<std::size_t>(in_idx)];
        }
    }
    return ours;
}

auto load_fused_cbr(FusedCBR2d& layer, std::ifstream& stream, const std::string& path) -> void
{
    const int out_channels = layer.out_channels();
    const int in_channels = layer.in_channels();
    const int kernel = layer.kernel_size();
    const std::size_t weights = static_cast<std::size_t>(out_channels) * static_cast<std::size_t>(in_channels)
        * static_cast<std::size_t>(kernel) * static_cast<std::size_t>(kernel);

    const auto beta = read_f32(stream, static_cast<std::size_t>(out_channels), path);
    const auto gamma = read_f32(stream, static_cast<std::size_t>(out_channels), path);
    const auto mean = read_f32(stream, static_cast<std::size_t>(out_channels), path);
    const auto var = read_f32(stream, static_cast<std::size_t>(out_channels), path);
    const auto filters = read_f32(stream, weights, path);

    const std::vector<int> channel_shape { 1, out_channels, 1, 1 };
    const std::vector<int> filter_shape { out_channels, in_channels, kernel, kernel };
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights", upload(filter_shape, filters));
    params.emplace("bias", zeros_like_bias(out_channels));
    params.emplace("gamma", upload(channel_shape, gamma));
    params.emplace("beta", upload(channel_shape, beta));
    params.emplace("running_mean", upload(channel_shape, mean));
    params.emplace("running_var", upload(channel_shape, var));
    layer.set_parameters(params);
}

auto load_conv(Conv2d& layer, std::ifstream& stream, const std::string& path, bool batch_normalize) -> void
{
    const int out_channels = layer.out_channels();
    const int in_channels = layer.in_channels();
    const int kernel = layer.kernel_size();
    const std::size_t weights = static_cast<std::size_t>(out_channels) * static_cast<std::size_t>(in_channels)
        * static_cast<std::size_t>(kernel) * static_cast<std::size_t>(kernel);
    const auto bias = read_f32(stream, static_cast<std::size_t>(out_channels), path);
    if (batch_normalize)
    {
        (void)read_f32(stream, static_cast<std::size_t>(out_channels) * 3U, path);
    }
    const auto filters = read_f32(stream, weights, path);
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights", upload({ out_channels, in_channels, kernel, kernel }, filters));
    params.emplace("bias", upload({ 1, out_channels, 1, 1 }, bias));
    layer.set_parameters(params);
}

auto load_local(LocalLayer& layer, std::ifstream& stream, const std::string& path) -> void
{
    const int locations = layer.locations();
    const std::size_t bias_count = static_cast<std::size_t>(layer.out_channels()) * static_cast<std::size_t>(locations);
    const std::size_t weight_count = bias_count * static_cast<std::size_t>(layer.in_channels())
        * static_cast<std::size_t>(layer.kernel_size()) * static_cast<std::size_t>(layer.kernel_size());
    const auto bias = read_f32(stream, bias_count, path);
    const auto weights = read_f32(stream, weight_count, path);
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights",
        upload({ locations, layer.out_channels(), layer.in_channels(), layer.kernel_size(), layer.kernel_size() },
            weights));
    params.emplace("bias", upload({ 1, layer.out_channels(), layer.out_height(), layer.out_width() }, bias));
    layer.set_parameters(params);
}

auto load_connected(FullyConnected& layer, std::ifstream& stream, const std::string& path, bool transpose_flag)
    -> void
{
    const int inputs = layer.input_size();
    const int outputs = layer.output_size();
    const auto bias = read_f32(stream, static_cast<std::size_t>(outputs), path);
    auto weights = read_f32(stream, static_cast<std::size_t>(inputs) * static_cast<std::size_t>(outputs), path);
    if (!transpose_flag)
    {
        weights = transpose_connected(weights, inputs, outputs);
    }
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights", upload({ inputs, outputs }, weights));
    params.emplace("bias", upload({ 1, outputs }, bias));
    layer.set_parameters(params);
}

} // namespace

auto load_darknet_weights(const std::vector<std::shared_ptr<Layer>>& layers, const std::string& path,
    const DarknetLoadOptions& options) -> DarknetLoadReport
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Could not open Darknet weights: " + path);
    }

    DarknetLoadReport report;
    read_exact(stream, &report.major, sizeof(int), path);
    read_exact(stream, &report.minor, sizeof(int), path);
    read_exact(stream, &report.revision, sizeof(int), path);
    if (((report.major * 10) + report.minor) >= 2 && report.major < 1000 && report.minor < 1000)
    {
        std::uint64_t seen = 0;
        read_exact(stream, &seen, sizeof(std::uint64_t), path);
        report.seen = seen;
    }
    else
    {
        int seen32 = 0;
        read_exact(stream, &seen32, sizeof(int), path);
        report.seen = static_cast<std::uint64_t>(seen32);
    }
    report.transpose_connected = (report.major > 1000) || (report.minor > 1000);

    int convs_seen = 0;
    for (const auto& layer : layers)
    {
        if (layer == nullptr)
        {
            continue;
        }
        if (auto* fused = dynamic_cast<FusedCBR2d*>(layer.get()))
        {
            if (options.cutoff_convs > 0 && convs_seen >= options.cutoff_convs)
            {
                continue;
            }
            load_fused_cbr(*fused, stream, path);
            ++convs_seen;
            ++report.convs_loaded;
            continue;
        }
        if (auto* conv = dynamic_cast<Conv2d*>(layer.get()))
        {
            if (options.cutoff_convs > 0 && convs_seen >= options.cutoff_convs)
            {
                continue;
            }
            load_conv(*conv, stream, path, false);
            ++convs_seen;
            ++report.convs_loaded;
            continue;
        }
        if (auto* local = dynamic_cast<LocalLayer*>(layer.get()))
        {
            if (!options.load_local)
            {
                continue;
            }
            load_local(*local, stream, path);
            ++report.locals_loaded;
            continue;
        }
        if (auto* fc = dynamic_cast<FullyConnected*>(layer.get()))
        {
            if (!options.load_connected)
            {
                continue;
            }
            load_connected(*fc, stream, path, report.transpose_connected);
            ++report.connected_loaded;
        }
    }

    report.bytes_remaining = remaining_bytes(stream);
    LOG_INFO("Darknet weights {}: version {}.{}.{} seen={} convs={} local={} fc={} leftover_bytes={}", path,
        report.major, report.minor, report.revision, report.seen, report.convs_loaded, report.locals_loaded,
        report.connected_loaded, report.bytes_remaining);
    return report;
}
