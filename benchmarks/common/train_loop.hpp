#pragma once

#include "run_metrics.hpp"

#include <chrono>

/**
 * The epoch skeleton shared by every training program.
 *
 * `prepare` returns the learning rate it just installed and leaves the model in train mode.
 * `train_split` walks the training rows and updates weights.
 * `eval_split` walks the held-out rows with updates off.
 * The clock covers all three, and the CSV row is the only record of the epoch.
 */
template <typename Prepare, typename TrainSplit, typename EvalSplit>
auto run_supervised_epochs(int total_epochs, std::ofstream& csv, Prepare prepare, TrainSplit train_split,
    EvalSplit eval_split, std::size_t (*vram_mib)() = nullptr) -> void
{
    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        const auto started = std::chrono::steady_clock::now();
        const float learning_rate = prepare(epoch);
        const SplitMetrics trained = train_split(epoch);
        const SplitMetrics tested = eval_split(epoch);
        const std::size_t vram = vram_mib != nullptr ? vram_mib() : current_vram_mib();
        write_split_row(csv, epoch, learning_rate, trained, tested, elapsed_seconds(started), vram);
    }
}
