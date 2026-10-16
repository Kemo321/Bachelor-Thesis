#pragma once

#include "DeepLearnLib/core/Batch.hpp"

#include <cstddef>
#include <future>
#include <random>
#include <string>
#include <utility>
#include <vector>

class ClassificationLoader
{
public:
    ClassificationLoader(std::string dataset_root, std::string split, int batch_size, int image_size = 32,
        bool shuffle = true, std::vector<std::string> class_names = {});
    ~ClassificationLoader();

    ClassificationLoader(const ClassificationLoader&) = delete;
    auto operator=(const ClassificationLoader&) -> ClassificationLoader& = delete;
    ClassificationLoader(ClassificationLoader&&) = delete;
    auto operator=(ClassificationLoader&&) -> ClassificationLoader& = delete;

    auto reset() -> void;
    [[nodiscard]] auto has_next() const -> bool;
    auto get_batch(cudaStream_t stream = 0) -> Batch;
    [[nodiscard]] auto size() const -> std::size_t;
    [[nodiscard]] auto batch_size() const -> int;
    [[nodiscard]] auto num_classes() const -> int;
    [[nodiscard]] auto image_size() const -> int;
    [[nodiscard]] auto class_names() const -> const std::vector<std::string>&;

private:
    struct HostBatch
    {
        int n { 0 };
        std::vector<float> images;
        std::vector<float> targets;
    };

    [[nodiscard]] auto load_sample(std::size_t sample_index, std::vector<float>& image_chw) const -> int;
    auto take_indices() -> std::vector<std::size_t>;
    [[nodiscard]] auto decode_indices(const std::vector<std::size_t>& indices) const -> HostBatch;
    auto upload_host_batch(HostBatch host, cudaStream_t stream) const -> Batch;
    auto launch_prefetch() -> void;
    auto join_prefetch() -> void;

    std::string split_root_;
    int batch_size_;
    int image_size_;
    bool shuffle_;
    std::vector<std::string> class_names_;
    std::vector<std::pair<std::string, int>> samples_;
    std::vector<std::size_t> order_;
    std::size_t cursor_ { 0 };
    std::mt19937 rng_;
    std::future<HostBatch> prefetch_;
};
