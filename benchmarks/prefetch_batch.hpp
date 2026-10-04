#pragma once

#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/dataset.hpp"

#include <functional>
#include <optional>

/**
 * Walk a loader with two CUDA streams.
 *
 * The CPU prefetch inside get_batch still runs. This loop additionally uploads
 * batch N+1 on the idle stream while batch N is trained on the other stream.
 * The callback must enqueue its GPU work on the stream it receives.
 */
template <typename Loader>
auto for_each_prefetched_batch(Loader& loader, const std::function<void(Batch&, int, cudaStream_t)>& step) -> int
{
    loader.reset();
    dl::UniqueCudaStream streams[2];
    std::optional<Batch> batches[2];
    bool ready[2] { false, false };
    if (loader.has_next())
    {
        batches[0] = loader.get_batch(streams[0].get());
        ready[0] = true;
    }

    int slot = 0;
    int count = 0;
    while (ready[slot])
    {
        const int next = 1 - slot;
        CHECK_CUDA(cudaStreamSynchronize(streams[slot].get()));
        const dl::StreamGuard stream_guard(streams[slot].get());
        step(*batches[slot], count, streams[slot].get());
        ++count;
        if (loader.has_next())
        {
            CHECK_CUDA(cudaStreamSynchronize(streams[next].get()));
            batches[next] = loader.get_batch(streams[next].get());
            ready[next] = true;
        }
        else
        {
            ready[next] = false;
            batches[next].reset();
        }
        slot = next;
    }
    CHECK_CUDA(cudaStreamSynchronize(streams[0].get()));
    CHECK_CUDA(cudaStreamSynchronize(streams[1].get()));
    return count;
}
