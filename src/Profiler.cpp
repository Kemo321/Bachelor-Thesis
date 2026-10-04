#include "DeepLearnLib/Profiler.hpp"

#include <stdexcept>
#include <string>

// A pair of CUDA events that times an interval on the GPU.
Profiler::Profiler()
{
#if DEEPLEARNLIB_ENABLE_CUDA
    CHECK_CUDA(cudaEventCreate(&start_event_));
    CHECK_CUDA(cudaEventCreate(&stop_event_));
#endif
}

// Destroys the events if they were created. A destroy error during shutdown is ignored.
Profiler::~Profiler()
{
#if DEEPLEARNLIB_ENABLE_CUDA
    if (start_event_ != nullptr)
    {
        static_cast<void>(cudaEventDestroy(start_event_));
        start_event_ = nullptr;
    }
    if (stop_event_ != nullptr)
    {
        static_cast<void>(cudaEventDestroy(stop_event_));
        stop_event_ = nullptr;
    }
#endif
}

// Records the start mark. Without CUDA there is no timeline, so this throws.
auto Profiler::start() -> void
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("Profiler::start requires CUDA");
#else
    CHECK_CUDA(cudaEventRecord(start_event_));
    running_ = true;
#endif
}

// Records the stop and returns milliseconds. The time between the events is valid only after the GPU synchronizes.
auto Profiler::stop() -> float
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("Profiler::stop requires CUDA");
#else
    if (!running_)
    {
        throw std::runtime_error("Profiler::stop requires a preceding start()");
    }
    CHECK_CUDA(cudaEventRecord(stop_event_));
    // The host may read the elapsed time only after the GPU reaches stop_event_.
    CHECK_CUDA(cudaEventSynchronize(stop_event_));
    float milliseconds = 0.0F;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start_event_, stop_event_));
    running_ = false;
    return milliseconds;
#endif
}

// Used MiB: (total - free) / 1024^2. When total < free, returns 0 instead of wrapping the subtraction.
auto Profiler::get_vram_usage_mb() -> std::size_t
{
#if !DEEPLEARNLIB_ENABLE_CUDA
    throw std::runtime_error("Profiler::get_vram_usage_mb requires CUDA");
#else
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (total_bytes < free_bytes)
    {
        // Unsigned subtraction would wrap, so report zero.
        return 0;
    }
    constexpr std::size_t kMib = 1024ULL * 1024ULL;
    return (total_bytes - free_bytes) / kMib;
#endif
}
