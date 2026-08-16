#pragma once

#include "DeepLearnLib/Tensor.hpp"

#include <cstddef>

/**
 * GPU wall-clock timing via cudaEvent and process VRAM readout.
 *
 * stop() is the first host sync; start() only records an event on the current stream.
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

    auto start() -> void;
    [[nodiscard]] auto stop() -> float;
    [[nodiscard]] static auto get_vram_usage_mb() -> std::size_t;

private:
#if DEEPLEARNLIB_ENABLE_CUDA
    cudaEvent_t start_event_ { nullptr };
    cudaEvent_t stop_event_ { nullptr };
#endif
    bool running_ { false };
};
