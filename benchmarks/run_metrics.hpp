#pragma once

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Profiler.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

/**
 * Shared epoch logging / CSV helpers so every train_* and inference_* binary
 * writes the same stdout columns.
 *
 * Epoch line (fields omitted when optional values are absent):
 *   TAG | Epoch [e/N] | LR: x | Train Loss: x | Test Loss: x | Train Acc: x |
 *        Test Acc: x | mAP@0.5: x | Batches: n | Time: ts | VRAM_MiB: v
 */
inline auto open_metrics_csv(const std::filesystem::path& results_dir, const std::string& filename,
    const std::string& header) -> std::ofstream
{
    std::filesystem::create_directories(results_dir);
    std::ofstream stream((results_dir / filename).string());
    if (!stream)
    {
        throw std::runtime_error("Failed to open metrics CSV: " + (results_dir / filename).string());
    }
    stream << header << "\n";
    return stream;
}

inline auto current_vram_mib() -> std::size_t
{
    return Profiler::get_vram_usage_mb();
}

inline auto accelerator_label() -> const char*
{
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0)
    {
        return "GPU";
    }
    return "CPU";
}

struct PipelineBanner
{
    const char* tag = "";
    const char* backend = "";
    int batch_size = 0;
    int epochs = 0;
    float learning_rate = 0.0F;
    float momentum = 0.0F;
    float weight_decay = 0.0F;
    float gradient_clip = 0.0F;
    std::string precision { "fp32" };
    int num_classes = 0;
    int workers = 0;
    int layers = 0;
    std::string dataset;
    std::size_t train_count = 0;
    std::size_t test_count = 0;
};

inline auto log_pipeline_banner(const PipelineBanner& banner) -> void
{
    LOG_INFO("{} | start backend={} device={}", banner.tag, banner.backend, accelerator_label());
    LOG_INFO("{} | config batch={} epochs={} lr={:.6g} momentum={:.4g} wd={:.6g} clip={:.4g} precision={} classes={} "
             "workers={}",
        banner.tag, banner.batch_size, banner.epochs, banner.learning_rate, banner.momentum, banner.weight_decay,
        banner.gradient_clip, banner.precision, banner.num_classes, banner.workers);
    LOG_INFO("{} | data root={} train={} test={} layers={}", banner.tag, banner.dataset, banner.train_count,
        banner.test_count, banner.layers);
    LOG_FLUSH();
}

struct EpochMetrics
{
    const char* tag = "";
    int epoch = 0;
    int total_epochs = 0;
    float lr = 0.0F;
    float train_loss = 0.0F;
    std::optional<float> test_loss;
    std::optional<float> train_acc;
    std::optional<float> test_acc;
    std::optional<float> map50;
    int train_batches = 0;
    long long time_s = 0;
    std::size_t vram_mib = 0;
};

inline auto log_train_epoch(const EpochMetrics& metrics) -> void
{
    std::ostringstream extra;
    extra.setf(std::ios::fixed);
    extra.precision(4);
    if (metrics.test_loss.has_value())
    {
        extra << " | Test Loss: " << *metrics.test_loss;
    }
    if (metrics.train_acc.has_value())
    {
        extra << " | Train Acc: " << *metrics.train_acc;
    }
    if (metrics.test_acc.has_value())
    {
        extra << " | Test Acc: " << *metrics.test_acc;
    }
    extra.precision(4);
    if (metrics.map50.has_value())
    {
        extra << " | mAP@0.5: " << *metrics.map50;
    }
    LOG_INFO("{} | Epoch [{}/{}] | LR: {:.6g} | Train Loss: {:.4f}{} | Batches: {} | Time: {}s | VRAM_MiB: {}",
        metrics.tag, metrics.epoch, metrics.total_epochs, metrics.lr, metrics.train_loss, extra.str(),
        metrics.train_batches, metrics.time_s, metrics.vram_mib);
    LOG_FLUSH();
}

inline auto log_train_epoch(const char* tag, int epoch, int total_epochs, float train_loss, float test_loss,
    long long time_s, std::size_t vram_mib) -> void
{
    log_train_epoch(EpochMetrics { tag, epoch, total_epochs, 0.0F, train_loss, test_loss, std::nullopt, std::nullopt,
        std::nullopt, 0, time_s, vram_mib });
}

inline auto log_train_epoch(const char* tag, int epoch, int total_epochs, float loss, long long time_s,
    std::size_t vram_mib) -> void
{
    log_train_epoch(EpochMetrics { tag, epoch, total_epochs, 0.0F, loss, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 0, time_s, vram_mib });
}

inline auto log_saved(const char* tag, const std::string& path) -> void
{
    LOG_INFO("{} | saved path={}", tag, path);
    LOG_FLUSH();
}

inline auto log_inference_start(const char* tag, const char* backend, const std::string& model, std::size_t images,
    float conf_threshold, float nms_threshold, const std::string& out_dir) -> void
{
    LOG_INFO("{} | start backend={} device={} model={} images={} conf={:.3g} nms={:.3g} out={}", tag, backend,
        accelerator_label(), model, images, conf_threshold, nms_threshold, out_dir);
    LOG_FLUSH();
}

inline auto log_inference_done(const char* tag, std::size_t saved, const std::string& out_dir) -> void
{
    LOG_INFO("{} | done saved={} out={}", tag, saved, out_dir);
    LOG_FLUSH();
}

inline auto write_train_test_row(std::ofstream& csv, int epoch, float train_loss, float test_loss, long long time_s,
    std::size_t vram_mib) -> void
{
    csv << epoch << ";" << train_loss << ";" << test_loss << ";" << time_s << ";" << vram_mib << "\n";
    csv.flush();
}

inline auto write_loss_row(std::ofstream& csv, int epoch, float loss, long long time_s, std::size_t vram_mib) -> void
{
    csv << epoch << ";" << loss << ";" << time_s << ";" << vram_mib << "\n";
    csv.flush();
}
