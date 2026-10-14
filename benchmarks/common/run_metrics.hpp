#pragma once

#include "DeepLearnLib/utils/Profiler.hpp"
#include "split_metrics.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

/**
 * One CSV schema for every training program, custom and Torch.
 *
 * Epoch;LR;TrainLoss;TestLoss;TrainAcc;TestAcc;Time(s);VRAM_MiB
 *
 * Time(s) is wall time of the epoch, including the test pass, written with
 * microsecond resolution. The decimal point is always '.'.
 */
inline constexpr const char* kMetricsCsvHeader = "Epoch;LR;TrainLoss;TestLoss;TrainAcc;TestAcc;Time(s);VRAM_MiB";

inline constexpr int kMetricDecimals = 6;

inline auto format_decimal(double value, int digits = kMetricDecimals) -> std::string
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out.setf(std::ios::fixed);
    out.precision(digits);
    out << value;
    return out.str();
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
    stream.imbue(std::locale::classic());
    stream << header << "\n";
    return stream;
}

inline auto current_vram_mib() -> std::size_t
{
    return Profiler::get_vram_usage_mb();
}

inline auto write_metrics_row(std::ofstream& csv, int epoch, float learning_rate, float train_loss, float test_loss,
    float train_acc, float test_acc, double time_s, std::size_t vram_mib) -> void
{
    csv << epoch << ";" << format_decimal(learning_rate) << ";" << format_decimal(train_loss) << ";"
        << format_decimal(test_loss) << ";" << format_decimal(train_acc) << ";" << format_decimal(test_acc) << ";"
        << format_decimal(time_s) << ";" << vram_mib << "\n";
    csv.flush();
}

inline auto write_split_row(std::ofstream& csv, int epoch, float learning_rate, const SplitMetrics& train,
    const SplitMetrics& test, double time_s, std::size_t vram_mib) -> void
{
    write_metrics_row(csv, epoch, learning_rate, train.loss(), test.loss(), train.accuracy(), test.accuracy(), time_s,
        vram_mib);
}

inline auto elapsed_seconds(std::chrono::steady_clock::time_point started) -> double
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}
