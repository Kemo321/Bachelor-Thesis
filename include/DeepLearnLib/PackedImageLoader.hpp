#pragma once

#include "DeepLearnLib/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <random>
#include <string>
#include <vector>

/**
 * Mini-batch loader for the packed `DLIMG001` classification format.
 *
 * File layout (little-endian):
 *   bytes 0-7   magic "DLIMG001"
 *   u32 n, channels, height, width, num_classes
 *   u8  pixels[n * channels * height * width]  (NCHW)
 *   u8  labels[n]
 *
 * Yields Batch tensors: images NCHW float in [0, 1], one-hot targets [N, C].
 * The whole split is kept in host RAM (MNIST train is ~47 MiB) so JPEG decode
 * is not on the hot path. Prefetch overlaps uint8→float conversion with GPU work.
 */
class PackedImageLoader
{
public:
    PackedImageLoader(std::string bin_path, int batch_size, bool shuffle = true);
    ~PackedImageLoader();

    PackedImageLoader(const PackedImageLoader&) = delete;
    auto operator=(const PackedImageLoader&) -> PackedImageLoader& = delete;
    PackedImageLoader(PackedImageLoader&&) = delete;
    auto operator=(PackedImageLoader&&) -> PackedImageLoader& = delete;

    auto reset() -> void;
    [[nodiscard]] auto has_next() const -> bool;
    auto get_batch(cudaStream_t stream = 0) -> Batch;
    [[nodiscard]] auto size() const -> std::size_t;
    [[nodiscard]] auto batch_size() const -> int;
    [[nodiscard]] auto num_classes() const -> int;
    [[nodiscard]] auto channels() const -> int;
    [[nodiscard]] auto height() const -> int;
    [[nodiscard]] auto width() const -> int;
    [[nodiscard]] auto class_names() const -> const std::vector<std::string>&;
    [[nodiscard]] auto label_at(std::size_t index) const -> int;
    auto copy_sample_float(std::size_t index, std::vector<float>& nchw) const -> void;

private:
    struct HostBatch
    {
        int n { 0 };
        std::vector<float> images;
        std::vector<float> targets;
    };

    auto take_indices() -> std::vector<std::size_t>;
    [[nodiscard]] auto decode_indices(const std::vector<std::size_t>& indices) const -> HostBatch;
    auto upload_host_batch(HostBatch host, cudaStream_t stream) const -> Batch;
    auto launch_prefetch() -> void;
    auto join_prefetch() -> void;

    int batch_size_;
    bool shuffle_;
    std::uint32_t sample_count_ { 0 };
    std::uint32_t channels_ { 0 };
    std::uint32_t height_ { 0 };
    std::uint32_t width_ { 0 };
    std::uint32_t num_classes_ { 0 };
    std::vector<std::uint8_t> pixels_;
    std::vector<std::uint8_t> labels_;
    std::vector<std::string> class_names_;
    std::vector<std::size_t> order_;
    std::size_t cursor_ { 0 };
    std::mt19937 rng_;
    std::future<HostBatch> prefetch_;
};
