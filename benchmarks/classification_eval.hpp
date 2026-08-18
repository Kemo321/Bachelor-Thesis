#pragma once

#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

/**
 * Shared classification eval helpers (argmax, accuracy, confusion CSV).
 * Vision sample dumps live in classification_vis.hpp (OpenCV).
 */
inline auto classification_argmax_row(const std::vector<float>& values, int row, int cols) -> int
{
    const std::size_t offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    int best = 0;
    float best_value = values[offset];
    for (int col = 1; col < cols; ++col)
    {
        const float value = values[offset + static_cast<std::size_t>(col)];
        if (value > best_value)
        {
            best_value = value;
            best = col;
        }
    }
    return best;
}

inline auto batch_accuracy_one_hot(const dl::Tensor& logits, const dl::Tensor& one_hot, cudaStream_t stream = 0)
    -> float
{
    const std::vector<float> logit_host = logits.to_host(stream);
    const std::vector<float> target_host = one_hot.to_host(stream);
    const int batch = logits.get_shape()[0];
    const int classes = logits.get_shape()[1];
    int correct = 0;
    for (int row = 0; row < batch; ++row)
    {
        if (classification_argmax_row(logit_host, row, classes)
            == classification_argmax_row(target_host, row, classes))
        {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(std::max(1, batch));
}

inline auto accumulate_confusion_one_hot(std::vector<int>& matrix, const dl::Tensor& logits, const dl::Tensor& one_hot,
    int num_classes, cudaStream_t stream = 0) -> void
{
    const std::vector<float> logit_host = logits.to_host(stream);
    const std::vector<float> target_host = one_hot.to_host(stream);
    const int batch = logits.get_shape()[0];
    matrix.resize(static_cast<std::size_t>(num_classes) * static_cast<std::size_t>(num_classes), 0);
    for (int row = 0; row < batch; ++row)
    {
        const int pred = classification_argmax_row(logit_host, row, num_classes);
        const int truth = classification_argmax_row(target_host, row, num_classes);
        ++matrix[(static_cast<std::size_t>(truth) * static_cast<std::size_t>(num_classes))
            + static_cast<std::size_t>(pred)];
    }
}

inline auto write_confusion_csv(const std::filesystem::path& path, const std::vector<int>& matrix, int num_classes,
    const std::vector<std::string>& class_names) -> void
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Could not write confusion CSV: " + path.string());
    }
    stream << "true\\pred";
    for (int col = 0; col < num_classes; ++col)
    {
        const std::string name = (col < static_cast<int>(class_names.size())) ? class_names[static_cast<std::size_t>(col)]
                                                                              : std::to_string(col);
        stream << ";" << name;
    }
    stream << "\n";
    for (int row = 0; row < num_classes; ++row)
    {
        const std::string name = (row < static_cast<int>(class_names.size())) ? class_names[static_cast<std::size_t>(row)]
                                                                             : std::to_string(row);
        stream << name;
        for (int col = 0; col < num_classes; ++col)
        {
            stream << ";"
                   << matrix[(static_cast<std::size_t>(row) * static_cast<std::size_t>(num_classes))
                       + static_cast<std::size_t>(col)];
        }
        stream << "\n";
    }
}

inline auto accumulate_confusion_ids(std::vector<int>& matrix, const std::vector<int>& truths,
    const std::vector<int>& preds, int num_classes) -> void
{
    matrix.resize(static_cast<std::size_t>(num_classes) * static_cast<std::size_t>(num_classes), 0);
    const std::size_t count = std::min(truths.size(), preds.size());
    for (std::size_t row = 0; row < count; ++row)
    {
        const int truth = std::clamp(truths[row], 0, num_classes - 1);
        const int pred = std::clamp(preds[row], 0, num_classes - 1);
        ++matrix[(static_cast<std::size_t>(truth) * static_cast<std::size_t>(num_classes))
            + static_cast<std::size_t>(pred)];
    }
}

inline auto write_class_names(const std::filesystem::path& path, const std::vector<std::string>& class_names) -> void
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Could not write class names: " + path.string());
    }
    for (const auto& name : class_names)
    {
        stream << name << "\n";
    }
}

struct SamplePrediction
{
    int index { 0 };
    int truth { 0 };
    int pred { 0 };
    float confidence { 0.0F };
    std::vector<float> image_nchw;
    int channels { 0 };
    int height { 0 };
    int width { 0 };
};

inline auto append_samples_from_ids(const dl::Tensor& images, const std::vector<int>& truths,
    const std::vector<int>& preds, const std::vector<float>& confidences, int start_index, int max_keep,
    std::vector<SamplePrediction>& out, cudaStream_t stream = 0) -> void
{
    if (static_cast<int>(out.size()) >= max_keep)
    {
        return;
    }
    const auto& image_shape = images.get_shape();
    const int batch = image_shape[0];
    const int channels = image_shape[1];
    const int height = image_shape[2];
    const int width = image_shape[3];
    const std::size_t elems = static_cast<std::size_t>(channels) * static_cast<std::size_t>(height)
        * static_cast<std::size_t>(width);
    const std::vector<float> image_host = images.to_host(stream);
    for (int row = 0; row < batch && static_cast<int>(out.size()) < max_keep; ++row)
    {
        SamplePrediction sample;
        sample.index = start_index + row;
        sample.truth = (row < static_cast<int>(truths.size())) ? truths[static_cast<std::size_t>(row)] : 0;
        sample.pred = (row < static_cast<int>(preds.size())) ? preds[static_cast<std::size_t>(row)] : 0;
        sample.confidence = (row < static_cast<int>(confidences.size())) ? confidences[static_cast<std::size_t>(row)]
                                                                        : 0.0F;
        sample.channels = channels;
        sample.height = height;
        sample.width = width;
        sample.image_nchw.assign(image_host.begin() + static_cast<std::ptrdiff_t>(row) * static_cast<std::ptrdiff_t>(elems),
            image_host.begin() + static_cast<std::ptrdiff_t>(row + 1) * static_cast<std::ptrdiff_t>(elems));
        out.push_back(std::move(sample));
    }
}

inline auto collect_batch_predictions(const dl::Tensor& images, const dl::Tensor& logits, const dl::Tensor& one_hot,
    int start_index, int max_keep, std::vector<SamplePrediction>& out, cudaStream_t stream = 0) -> void
{
    if (static_cast<int>(out.size()) >= max_keep)
    {
        return;
    }
    const auto& image_shape = images.get_shape();
    const int batch = logits.get_shape()[0];
    const int classes = logits.get_shape()[1];
    const int channels = image_shape[1];
    const int height = image_shape[2];
    const int width = image_shape[3];
    const std::size_t elems = static_cast<std::size_t>(channels) * static_cast<std::size_t>(height)
        * static_cast<std::size_t>(width);
    const std::vector<float> image_host = images.to_host(stream);
    const std::vector<float> logit_host = logits.to_host(stream);
    const std::vector<float> target_host = one_hot.to_host(stream);
    for (int row = 0; row < batch && static_cast<int>(out.size()) < max_keep; ++row)
    {
        SamplePrediction sample;
        sample.index = start_index + row;
        sample.pred = classification_argmax_row(logit_host, row, classes);
        sample.truth = classification_argmax_row(target_host, row, classes);
        float max_logit = logit_host[(static_cast<std::size_t>(row) * static_cast<std::size_t>(classes))
            + static_cast<std::size_t>(sample.pred)];
        float sum_exp = 0.0F;
        for (int col = 0; col < classes; ++col)
        {
            const float logit = logit_host[(static_cast<std::size_t>(row) * static_cast<std::size_t>(classes))
                + static_cast<std::size_t>(col)];
            sum_exp += std::exp(logit - max_logit);
        }
        sample.confidence = 1.0F / std::max(sum_exp, 1.0e-8F);
        sample.channels = channels;
        sample.height = height;
        sample.width = width;
        sample.image_nchw.assign(image_host.begin() + static_cast<std::ptrdiff_t>(row) * static_cast<std::ptrdiff_t>(elems),
            image_host.begin() + static_cast<std::ptrdiff_t>(row + 1) * static_cast<std::ptrdiff_t>(elems));
        out.push_back(std::move(sample));
    }
}

inline auto write_predictions_csv(const std::filesystem::path& path, const std::vector<SamplePrediction>& samples,
    const std::vector<std::string>& class_names) -> void
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Could not write predictions CSV: " + path.string());
    }
    stream << "index;true;pred;correct;confidence\n";
    for (const auto& sample : samples)
    {
        const std::string truth_name = (sample.truth < static_cast<int>(class_names.size()))
            ? class_names[static_cast<std::size_t>(sample.truth)]
            : std::to_string(sample.truth);
        const std::string pred_name = (sample.pred < static_cast<int>(class_names.size()))
            ? class_names[static_cast<std::size_t>(sample.pred)]
            : std::to_string(sample.pred);
        stream << sample.index << ";" << truth_name << ";" << pred_name << ";" << (sample.truth == sample.pred ? 1 : 0)
               << ";" << sample.confidence << "\n";
    }
}

template <typename Loader>
auto for_each_prefetched_batch(Loader& loader, const std::function<void(Batch&, int, cudaStream_t)>& step) -> int
{
    loader.reset();
    dl::UniqueCudaStream streams[2];
    std::optional<Batch> batches[2];
    bool ready[2] { false, false };
    if (loader.has_next())
    {
        batches[0] = loader.get_batch(streams[0].get());
        ready[0] = true;
    }

    int slot = 0;
    int count = 0;
    while (ready[slot])
    {
        const int next = 1 - slot;
        CHECK_CUDA(cudaStreamSynchronize(streams[slot].get()));
        const dl::StreamGuard stream_guard(streams[slot].get());
        step(*batches[slot], count, streams[slot].get());
        ++count;
        if (loader.has_next())
        {
            CHECK_CUDA(cudaStreamSynchronize(streams[next].get()));
            batches[next] = loader.get_batch(streams[next].get());
            ready[next] = true;
        }
        else
        {
            ready[next] = false;
            batches[next].reset();
        }
        slot = next;
    }
    CHECK_CUDA(cudaStreamSynchronize(streams[0].get()));
    CHECK_CUDA(cudaStreamSynchronize(streams[1].get()));
    return count;
}
