#pragma once

#include "DeepLearnLib/Tensor.hpp"

#include <cstddef>

/**
 * @brief GPU wall-clock timing via cudaEvent and process VRAM readout.
 *
 * The sequential graph has no autograd tape; this profiler measures kernels
 * that actually run on the device without introducing extra host syncs until
 * stop() is called.
 */
class Profiler
{
public:
    Profiler();
    ~Profiler();

    Profiler(const Profiler&) = delete;
    auto operator=(const Profiler&) -> Profiler& = delete;
    Profiler(Profiler&&) = delete;
    auto operator=(Profiler&&) -> Profiler& = delete;

    /**
     * @brief Records a start CUDA event on the current stream.
     */
    auto start() -> void;

    /**
     * @brief Records a stop event, synchronizes it, and returns elapsed milliseconds.
     */
    [[nodiscard]] auto stop() -> float;

    /**
     * @brief Bytes currently allocated on the device, in mebibytes (total - free).
     */
    [[nodiscard]] static auto get_vram_usage_mb() -> std::size_t;

private:
#if DEEPLEARNLIB_ENABLE_CUDA
    cudaEvent_t start_event_ { nullptr };
    cudaEvent_t stop_event_ { nullptr };
#endif
    bool running_ { false };
};
