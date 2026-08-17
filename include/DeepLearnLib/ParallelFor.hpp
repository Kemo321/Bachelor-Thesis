#pragma once

#include <algorithm>
#include <exception>
#include <future>
#include <thread>
#include <vector>

namespace dl
{

inline auto parallel_worker_count(int work_items) -> int
{
    if (work_items <= 1)
    {
        return 1;
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    int cap = (hardware == 0U) ? 8 : static_cast<int>(hardware);
    cap = std::min(cap, 16);
    return std::min(work_items, cap);
}

/**
 * Run `fn(index)` for index in [0, count) on a bounded thread pool.
 *
 * Strides work across workers so JPEG decode of a mini-batch does not spawn
 * one OS thread per sample (which thrashes at CIFAR batch 64).
 */
template <typename Fn>
auto parallel_for(int count, Fn&& fn) -> void
{
    if (count <= 0)
    {
        return;
    }
    if (count == 1)
    {
        fn(0);
        return;
    }

    const int workers = parallel_worker_count(count);
    std::vector<std::future<void>> tasks;
    tasks.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker)
    {
        tasks.emplace_back(std::async(std::launch::async,
            [worker, workers, count, &fn]()
            {
                for (int index = worker; index < count; index += workers)
                {
                    fn(index);
                }
            }));
    }

    std::exception_ptr first_error;
    for (auto& task : tasks)
    {
        try
        {
            task.get();
        }
        catch (...)
        {
            if (!first_error)
            {
                first_error = std::current_exception();
            }
        }
    }
    if (first_error)
    {
        std::rethrow_exception(first_error);
    }
}

} // namespace dl
