#pragma once

#include "DeepLearnLib/Profiler.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

inline constexpr int kMicroBatch = 16;
inline constexpr int kMicroImage = 448;
inline constexpr int kMicroChannels = 3;
inline constexpr int kMicroWarmup = 5;
inline constexpr int kMicroModelWarmup = 2;

inline auto micro_require_cuda(benchmark::State& state) -> bool
{
    if (!torch::cuda::is_available())
    {
        state.SkipWithError("CUDA is required for micro-benchmarks");
        return false;
    }
    return true;
}

inline auto micro_numel(const std::vector<int>& shape) -> std::size_t
{
    std::size_t count = 1;
    for (int dimension : shape)
    {
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

inline auto micro_host_filled(const std::vector<int>& shape, float value) -> std::vector<float>
{
    return std::vector<float>(micro_numel(shape), value);
}

inline auto micro_gpu_tensor(const std::vector<int>& shape, float value) -> dl::Tensor
{
    return dl::Tensor::from_host(shape, micro_host_filled(shape, value), dl::Device::GPU);
}

inline auto micro_torch_cuda(at::IntArrayRef shape, float value, bool requires_grad = false) -> torch::Tensor
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto tensor = torch::full(shape, value, options);
    if (requires_grad)
    {
        tensor.requires_grad_(true);
    }
    return tensor;
}

template <typename Body>
inline auto micro_gpu_loop(benchmark::State& state, std::size_t bytes, Body&& body) -> void
{
    Profiler profiler;
    for (auto _ : state)
    {
        profiler.start();
        body();
        const float milliseconds = profiler.stop();
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(bytes));
    state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
}

template <typename LayerT>
inline auto micro_custom_fwd(benchmark::State& state, LayerT& layer, const dl::Tensor& input) -> void
{
    layer.to(dl::Device::GPU);
    dl::Tensor output;
    for (int index = 0; index < kMicroWarmup; ++index)
    {
        output = layer.forward(input);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    micro_gpu_loop(state, micro_numel(input.get_shape()) * sizeof(float),
        [&]
        {
            output = layer.forward(input);
        });
}

template <typename LayerT>
inline auto micro_custom_bwd(benchmark::State& state, LayerT& layer, const dl::Tensor& input) -> void
{
    layer.to(dl::Device::GPU);
    dl::Tensor output = layer.forward(input);
    for (int index = 1; index < kMicroWarmup; ++index)
    {
        output = layer.forward(input);
    }
    const dl::Tensor grad_out = micro_gpu_tensor(output.get_shape(), 1.0F);
    dl::Tensor grad_in = layer.backward(grad_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    const std::size_t bytes = micro_numel(output.get_shape()) * sizeof(float);
    Profiler profiler;
    for (auto _ : state)
    {
        output = layer.forward(input);
        CHECK_CUDA(cudaDeviceSynchronize());
        profiler.start();
        grad_in = layer.backward(grad_out);
        const float milliseconds = profiler.stop();
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(bytes));
    state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
}
