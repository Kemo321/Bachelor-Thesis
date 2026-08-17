#pragma once

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Profiler.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

/**
 * Shared epoch logging / CSV helpers so every train_* binary writes the same
 * columns: Epoch, Loss (or TrainLoss/TestLoss), Time(s), VRAM_MiB.
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

inline auto log_train_epoch(const char* tag, int epoch, int total_epochs, float train_loss, float test_loss,
    long long time_s, std::size_t vram_mib) -> void
{
    LOG_INFO("{} | Epoch [{}/{}] | Loss: {:.4f} | Train Loss: {:.4f} | Test Loss: {:.4f} | Time: {}s | VRAM_MiB: {}",
        tag, epoch, total_epochs, train_loss, train_loss, test_loss, time_s, vram_mib);
}

inline auto log_train_epoch(const char* tag, int epoch, int total_epochs, float loss, long long time_s,
    std::size_t vram_mib) -> void
{
    LOG_INFO("{} | Epoch [{}/{}] | Loss: {:.4f} | Time: {}s | VRAM_MiB: {}", tag, epoch, total_epochs, loss, time_s,
        vram_mib);
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
