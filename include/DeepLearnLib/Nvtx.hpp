#pragma once

#ifndef DEEPLEARNLIB_ENABLE_CUDA
#define DEEPLEARNLIB_ENABLE_CUDA 1
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#if DEEPLEARNLIB_ENABLE_CUDA
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DEEPLEARNLIB_HAS_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define DEEPLEARNLIB_HAS_NVTX 1
#endif
#endif
#endif

#ifndef DEEPLEARNLIB_HAS_NVTX
#define DEEPLEARNLIB_HAS_NVTX 0
#endif

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace dl
{

/**
 * RAII NVTX range for Nsight Systems timelines.
 *
 * Pushes `nvtxRangePushA` on construction and `nvtxRangePop` on destruction so
 * CUDA error throws cannot leave the marker stack unbalanced. Compiles to a
 * no-op when NVTX headers are unavailable.
 */
class NvtxRange
{
public:
    explicit NvtxRange(const char* name)
    {
#if DEEPLEARNLIB_HAS_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~NvtxRange()
    {
#if DEEPLEARNLIB_HAS_NVTX
        nvtxRangePop();
#endif
    }

    NvtxRange(const NvtxRange&) = delete;
    auto operator=(const NvtxRange&) -> NvtxRange& = delete;
    NvtxRange(NvtxRange&&) = delete;
    auto operator=(NvtxRange&&) -> NvtxRange& = delete;
};

} // namespace dl
