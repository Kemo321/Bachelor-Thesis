#pragma once

#include "DeepLearnLib/core/Tensor.hpp"

#include <cstddef>

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
