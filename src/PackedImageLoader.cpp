#include "DeepLearnLib/PackedImageLoader.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Nvtx.hpp"
#include "DeepLearnLib/ParallelFor.hpp"
#include "DeepLearnLib/Precision.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

constexpr std::array<char, 8> kMagic { 'D', 'L', 'I', 'M', 'G', '0', '0', '1' };

auto read_u32_le(const std::uint8_t* bytes) -> std::uint32_t
{
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

auto default_class_names(int num_classes) -> std::vector<std::string>
{
    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(num_classes));
    for (int class_id = 0; class_id < num_classes; ++class_id)
    {
        names.push_back(std::to_string(class_id));
    }
    return names;
}

} // namespace

PackedImageLoader::PackedImageLoader(std::string bin_path, int batch_size, bool shuffle)
    : batch_size_(batch_size)
    , shuffle_(shuffle)
    , rng_(std::random_device {}())
{
    if (batch_size_ <= 0)
    {
        throw std::runtime_error("PackedImageLoader requires a positive batch size");
    }

    std::ifstream stream(bin_path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("PackedImageLoader could not open: " + bin_path);
    }

    std::array<char, 8> magic {};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != kMagic)
    {
        throw std::runtime_error("PackedImageLoader expected DLIMG001 magic in " + bin_path);
    }

    std::array<std::uint8_t, 20> header {};
    stream.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    if (!stream)
    {
        throw std::runtime_error("PackedImageLoader truncated header in " + bin_path);
    }

    sample_count_ = read_u32_le(header.data());
    channels_ = read_u32_le(header.data() + 4);
    height_ = read_u32_le(header.data() + 8);
    width_ = read_u32_le(header.data() + 12);
    num_classes_ = read_u32_le(header.data() + 16);
    if (sample_count_ == 0 || channels_ == 0 || height_ == 0 || width_ == 0 || num_classes_ == 0)
    {
        throw std::runtime_error("PackedImageLoader header has a zero dimension in " + bin_path);
    }

    const std::size_t pixel_count = static_cast<std::size_t>(sample_count_) * static_cast<std::size_t>(channels_)
        * static_cast<std::size_t>(height_) * static_cast<std::size_t>(width_);
    pixels_.resize(pixel_count);
    labels_.resize(static_cast<std::size_t>(sample_count_));
    stream.read(reinterpret_cast<char*>(pixels_.data()), static_cast<std::streamsize>(pixel_count));
    stream.read(reinterpret_cast<char*>(labels_.data()), static_cast<std::streamsize>(labels_.size()));
    if (!stream)
    {
        throw std::runtime_error("PackedImageLoader truncated payload in " + bin_path);
    }

    class_names_ = default_class_names(static_cast<int>(num_classes_));
    LOG_DEBUG("PackedImageLoader {} n={} c={} h={} w={} classes={} batch={} shuffle={}", bin_path, sample_count_,
        channels_, height_, width_, num_classes_, batch_size_, shuffle_);
    reset();
}

PackedImageLoader::~PackedImageLoader()
{
    join_prefetch();
}

auto PackedImageLoader::reset() -> void
{
    join_prefetch();
    order_.resize(static_cast<std::size_t>(sample_count_));
    std::iota(order_.begin(), order_.end(), 0);
    if (shuffle_)
    {
        std::shuffle(order_.begin(), order_.end(), rng_);
    }
    cursor_ = 0;
    launch_prefetch();
}

auto PackedImageLoader::has_next() const -> bool
{
    return cursor_ < order_.size() || prefetch_.valid();
}

auto PackedImageLoader::size() const -> std::size_t
{
    return static_cast<std::size_t>(sample_count_);
}

auto PackedImageLoader::batch_size() const -> int
{
    return batch_size_;
}

auto PackedImageLoader::num_classes() const -> int
{
    return static_cast<int>(num_classes_);
}

auto PackedImageLoader::channels() const -> int
{
    return static_cast<int>(channels_);
}

auto PackedImageLoader::height() const -> int
{
    return static_cast<int>(height_);
}

auto PackedImageLoader::width() const -> int
{
    return static_cast<int>(width_);
}

auto PackedImageLoader::class_names() const -> const std::vector<std::string>&
{
    return class_names_;
}

auto PackedImageLoader::label_at(std::size_t index) const -> int
{
    if (index >= labels_.size())
    {
        throw std::runtime_error("PackedImageLoader::label_at index out of range");
    }
    return static_cast<int>(labels_[index]);
}

auto PackedImageLoader::copy_sample_float(std::size_t index, std::vector<float>& nchw) const -> void
{
    const std::size_t elems = static_cast<std::size_t>(channels_) * static_cast<std::size_t>(height_)
        * static_cast<std::size_t>(width_);
    if (index >= static_cast<std::size_t>(sample_count_))
    {
        throw std::runtime_error("PackedImageLoader::copy_sample_float index out of range");
    }
    nchw.resize(elems);
    const std::size_t offset = index * elems;
    for (std::size_t i = 0; i < elems; ++i)
    {
        nchw[i] = static_cast<float>(pixels_[offset + i]) / 255.0F;
    }
}

auto PackedImageLoader::take_indices() -> std::vector<std::size_t>
{
    std::vector<std::size_t> indices;
    if (cursor_ >= order_.size())
    {
        return indices;
    }
    const std::size_t remaining = order_.size() - cursor_;
    const int this_batch = static_cast<int>(std::min(remaining, static_cast<std::size_t>(batch_size_)));
    indices.reserve(static_cast<std::size_t>(this_batch));
    for (int batch_idx = 0; batch_idx < this_batch; ++batch_idx)
    {
        indices.push_back(order_[cursor_++]);
    }
    return indices;
}

auto PackedImageLoader::decode_indices(const std::vector<std::size_t>& indices) const -> HostBatch
{
    HostBatch host;
    host.n = static_cast<int>(indices.size());
    if (host.n == 0)
    {
        return host;
    }
    const std::size_t elems = static_cast<std::size_t>(channels_) * static_cast<std::size_t>(height_)
        * static_cast<std::size_t>(width_);
    const int classes = num_classes();
    host.images.assign(static_cast<std::size_t>(host.n) * elems, 0.0F);
    host.targets.assign(static_cast<std::size_t>(host.n) * static_cast<std::size_t>(classes), 0.0F);

    dl::parallel_for(host.n,
        [this, &indices, &host, elems, classes](int batch_idx)
        {
            const std::size_t sample_index = indices[static_cast<std::size_t>(batch_idx)];
            const std::size_t src = sample_index * elems;
            const std::size_t dst = static_cast<std::size_t>(batch_idx) * elems;
            for (std::size_t i = 0; i < elems; ++i)
            {
                host.images[dst + i] = static_cast<float>(pixels_[src + i]) / 255.0F;
            }
            const int label = std::clamp(static_cast<int>(labels_[sample_index]), 0, classes - 1);
            host.targets[(static_cast<std::size_t>(batch_idx) * static_cast<std::size_t>(classes))
                + static_cast<std::size_t>(label)]
                = 1.0F;
        });
    return host;
}

auto PackedImageLoader::upload_host_batch(HostBatch host, cudaStream_t stream) const -> Batch
{
    return Batch { dl::Tensor::from_host({ host.n, static_cast<int>(channels_), static_cast<int>(height_),
                       static_cast<int>(width_) },
                       host.images, dl::Device::GPU, stream, dl::compute_dtype()),
        dl::Tensor::from_host(
            { host.n, num_classes() }, host.targets, dl::Device::GPU, stream, dl::compute_dtype()) };
}

auto PackedImageLoader::launch_prefetch() -> void
{
    auto indices = take_indices();
    if (indices.empty())
    {
        return;
    }
    prefetch_ = std::async(std::launch::async, [this, indices = std::move(indices)]()
        { return decode_indices(indices); });
}

auto PackedImageLoader::join_prefetch() -> void
{
    if (!prefetch_.valid())
    {
        return;
    }
    try
    {
        prefetch_.wait();
    }
    catch (...)
    {
    }
    prefetch_ = {};
}

auto PackedImageLoader::get_batch(cudaStream_t stream) -> Batch
{
    const dl::NvtxRange nvtx_range("PackedImageLoader_GetBatch");
    if (!prefetch_.valid())
    {
        launch_prefetch();
    }
    if (!prefetch_.valid())
    {
        throw std::runtime_error("PackedImageLoader::get_batch called with no remaining samples");
    }

    HostBatch host = prefetch_.get();
    prefetch_ = {};
    launch_prefetch();
    if (host.n <= 0)
    {
        throw std::runtime_error("PackedImageLoader decoded an empty batch");
    }
    return upload_host_batch(std::move(host), stream);
}
