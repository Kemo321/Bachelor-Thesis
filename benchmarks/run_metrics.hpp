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
 * Shared epoch logging / CSV helpers.
 *
 * Every training line prints the same fields. A task that does not compute one
 * writes n/a on stdout. CSV headers stay numeric and are one of the three below.
 *
 * Detection:  Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;mAP@0.5
 * Classification: Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;TrainAcc;TestAcc
 * Tabular (no held-out split): Epoch;TrainLoss;Time(s);VRAM_MiB;TrainAcc
 */
inline constexpr const char* kDetectionCsvHeader = "Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;mAP@0.5";
inline constexpr const char* kClassificationCsvHeader = "Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;TrainAcc;TestAcc";
inline constexpr const char* kTabularCsvHeader = "Epoch;TrainLoss;Time(s);VRAM_MiB;TrainAcc";

// Info lines from this process, next to the CSV. A new run replaces the file.
inline auto open_results_log(const std::filesystem::path& results_dir, const std::string& filename) -> void
{
    dl::log_to_file(results_dir / filename);
}

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
    LOG_INFO("{} | config batch={} epochs={} lr={:.6g} momentum={:.4g} wd={:.6g} clip={:.4g} classes={} workers={}",
        banner.tag, banner.batch_size, banner.epochs, banner.learning_rate, banner.momentum, banner.weight_decay,
        banner.gradient_clip, banner.num_classes, banner.workers);
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

inline auto format_metric(const std::optional<float>& value) -> std::string
{
    if (!value.has_value())
    {
        return "n/a";
    }
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(4);
    out << *value;
    return out.str();
}

inline auto log_train_epoch(const EpochMetrics& metrics) -> void
{
    LOG_INFO("{} | Epoch [{}/{}] | LR: {:.6g} | Train Loss: {:.4f} | Test Loss: {} | Train Acc: {} | Test Acc: {} | "
             "mAP@0.5: {} | Batches: {} | Time: {}s | VRAM_MiB: {}",
        metrics.tag, metrics.epoch, metrics.total_epochs, metrics.lr, metrics.train_loss,
        format_metric(metrics.test_loss), format_metric(metrics.train_acc), format_metric(metrics.test_acc),
        format_metric(metrics.map50), metrics.train_batches, metrics.time_s, metrics.vram_mib);
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

inline auto write_detection_row(std::ofstream& csv, int epoch, float train_loss, float test_loss, long long time_s,
    std::size_t vram_mib, float map50) -> void
{
    csv << epoch << ";" << train_loss << ";" << test_loss << ";" << time_s << ";" << vram_mib << ";" << map50 << "\n";
    csv.flush();
}

inline auto write_classification_row(std::ofstream& csv, int epoch, float train_loss, float test_loss, long long time_s,
    std::size_t vram_mib, float train_acc, float test_acc) -> void
{
    csv << epoch << ";" << train_loss << ";" << test_loss << ";" << time_s << ";" << vram_mib << ";" << train_acc << ";"
        << test_acc << "\n";
    csv.flush();
}

inline auto write_tabular_row(std::ofstream& csv, int epoch, float train_loss, long long time_s, std::size_t vram_mib,
    float train_acc) -> void
{
    csv << epoch << ";" << train_loss << ";" << time_s << ";" << vram_mib << ";" << train_acc << "\n";
    csv.flush();
}
