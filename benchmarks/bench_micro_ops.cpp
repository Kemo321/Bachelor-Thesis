#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/Profiler.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "TorchYOLO.hpp"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// YOLOv1 image transfers, the 64→192 convolution, the first head FC layer, and the 7×7 grid loss. Manual time is the Profiler GPU interval, not the Google Benchmark wall clock; the unit is milliseconds.
// Shapes at batch 16: image 3×448, map 112×112, FC 7×7×1024→4096, cell depth 10+20.

namespace
{

constexpr int kBatch = 16;
constexpr int kImage = 448;
constexpr int kChannels = 3;
constexpr int kConvIn = 64;
constexpr int kConvOut = 192;
constexpr int kSpatial = 112;
constexpr int kKernel = 3;
constexpr int kFcIn = 7 * 7 * 1024;
constexpr int kFcOut = 4096;
constexpr int kGrid = 7;
constexpr int kYoloClasses = 20;
constexpr int kYoloDepth = 10 + kYoloClasses;
constexpr int kWarmup = 5;

// Skips the case when the process cannot see CUDA.
auto require_cuda(benchmark::State& state) -> bool
{
    if (!torch::cuda::is_available())
    {
        state.SkipWithError("CUDA is required for micro-benchmarks");
        return false;
    }
    return true;
}

// Element count of a tensor with the given shape.
auto numel(const std::vector<int>& shape) -> std::size_t
{
    std::size_t count = 1;
    for (int dimension : shape)
    {
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

// Host buffer filled with a constant, for upload to the GPU.
auto host_filled(const std::vector<int>& shape, float value) -> std::vector<float>
{
    return std::vector<float>(numel(shape), value);
}

// Google Benchmark loop: manual time from the Profiler, a byte counter, and occupied VRAM.
template <typename Body>
auto run_gpu_loop(benchmark::State& state, std::size_t bytes, Body&& body) -> void
{
    Profiler profiler;
    for (auto _ : state)
    {
        profiler.start();
        body();
        const float milliseconds = profiler.stop();
        // Iteration time is the Profiler GPU interval, not the Google Benchmark wall clock.
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(bytes));
    state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
}

// A few Conv2d passes before measurement, then a device synchronization.
auto warmup_custom_conv(Conv2d& conv, const dl::Tensor& input) -> dl::Tensor
{
    dl::Tensor output;
    for (int index = 0; index < kWarmup; ++index)
    {
        output = conv.forward(input);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return output;
}

} // namespace

// BM_H2D_Custom_FromHost / BM_H2D_Torch_To: upload of the image [16, 3, 448, 448] including allocation; Custom uses Tensor::from_host, LibTorch uses to(CUDA).
static void BM_H2D_Custom_FromHost(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kBatch, kChannels, kImage, kImage };
    const std::vector<float> host = host_filled(shape, 0.25F);
    const std::size_t bytes = numel(shape) * sizeof(float);
    {
        dl::Tensor warm = dl::Tensor::from_host(shape, host, dl::Device::GPU);
        (void)warm;
    }

    dl::Tensor uploaded;
    run_gpu_loop(state, bytes,
        [&]
        {
            uploaded = dl::Tensor::from_host(shape, host, dl::Device::GPU);
        });
}

static void BM_H2D_Torch_To(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    auto cpu = torch::full({ kBatch, kChannels, kImage, kImage }, 0.25F, torch::kFloat32).contiguous();
    const std::size_t bytes = static_cast<std::size_t>(cpu.numel()) * sizeof(float);
    {
        auto warm = cpu.to(torch::kCUDA);
        torch::cuda::synchronize();
        (void)warm;
    }

    torch::Tensor uploaded;
    run_gpu_loop(state, bytes,
        [&]
        {
            uploaded = cpu.to(torch::kCUDA);
            torch::cuda::synchronize();
        });
}

// BM_H2D_Custom_ReuseMemcpy / BM_H2D_Torch_CopyInto: the same 16×3×448 image into an existing buffer; Custom uses cudaMemcpyAsync from pinned memory, LibTorch uses copy_.
static void BM_H2D_Custom_ReuseMemcpy(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kBatch, kChannels, kImage, kImage };
    const std::size_t bytes = numel(shape) * sizeof(float);
    const std::vector<float> host = host_filled(shape, 0.25F);
    dl::Tensor gpu(shape, dl::Device::GPU);
    float* pinned { nullptr };
    CHECK_CUDA(cudaMallocHost(&pinned, bytes));
    std::memcpy(pinned, host.data(), bytes);
    CHECK_CUDA(cudaMemcpy(gpu.data(), pinned, bytes, cudaMemcpyHostToDevice));

    run_gpu_loop(state, bytes,
        [&]
        {
            CHECK_CUDA(cudaMemcpyAsync(gpu.data(), pinned, bytes, cudaMemcpyHostToDevice, dl::current_stream()));
        });
    CHECK_CUDA(cudaFreeHost(pinned));
}

static void BM_H2D_Torch_CopyInto(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    auto cpu = torch::full({ kBatch, kChannels, kImage, kImage }, 0.25F, torch::kFloat32).contiguous();
    auto gpu = torch::empty_like(cpu, cpu.options().device(torch::kCUDA));
    gpu.copy_(cpu);
    torch::cuda::synchronize();
    const std::size_t bytes = static_cast<std::size_t>(cpu.numel()) * sizeof(float);

    run_gpu_loop(state, bytes,
        [&]
        {
            gpu.copy_(cpu);
            torch::cuda::synchronize();
        });
}

// BM_D2H_Custom_ToHost / BM_D2H_Torch_Cpu: read of the image [16, 3, 448, 448] from GPU to host; Custom uses to_host, LibTorch uses cpu().
static void BM_D2H_Custom_ToHost(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kBatch, kChannels, kImage, kImage };
    const std::vector<float> host = host_filled(shape, 0.25F);
    const dl::Tensor gpu = dl::Tensor::from_host(shape, host, dl::Device::GPU);
    const std::size_t bytes = numel(shape) * sizeof(float);
    (void)gpu.to_host();

    std::vector<float> downloaded;
    run_gpu_loop(state, bytes,
        [&]
        {
            downloaded = gpu.to_host();
        });
}

static void BM_D2H_Torch_Cpu(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    auto gpu = torch::full({ kBatch, kChannels, kImage, kImage }, 0.25F, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    torch::cuda::synchronize();
    const std::size_t bytes = static_cast<std::size_t>(gpu.numel()) * sizeof(float);
    (void)gpu.cpu();

    torch::Tensor downloaded;
    run_gpu_loop(state, bytes,
        [&]
        {
            downloaded = gpu.cpu();
            torch::cuda::synchronize();
        });
}

// BM_Conv2d_Fwd_Custom / BM_Conv2d_Fwd_Torch: convolution 64→192, 3×3 kernel, padding 1, map [16, 64, 112, 112] (second YOLOv1 block).
static void BM_Conv2d_Fwd_Custom(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    // Conv shape: batch 16, 64→192, kernel 3, spatial 112.
    Conv2d conv(kConvIn, kConvOut, kKernel, 1, 1);
    conv.to(dl::Device::GPU);
    const std::vector<int> shape { kBatch, kConvIn, kSpatial, kSpatial };
    const dl::Tensor input = dl::Tensor::from_host(shape, host_filled(shape, 0.1F), dl::Device::GPU);
    dl::Tensor output = warmup_custom_conv(conv, input);
    const std::size_t bytes = numel(shape) * sizeof(float);

    run_gpu_loop(state, bytes,
        [&]
        {
            output = conv.forward(input);
        });
}

static void BM_Conv2d_Fwd_Torch(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    at::globalContext().setBenchmarkCuDNN(true);
    // Conv shape: batch 16, 64→192, kernel 3, spatial 112.
    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(kConvIn, kConvOut, kKernel).padding(1));
    conv->to(torch::kCUDA);
    conv->eval();
    auto input = torch::full({ kBatch, kConvIn, kSpatial, kSpatial }, 0.1F, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        for (int index = 0; index < kWarmup; ++index)
        {
            output = conv->forward(input);
        }
        torch::cuda::synchronize();
    }
    const std::size_t bytes = static_cast<std::size_t>(input.numel()) * sizeof(float);

    torch::NoGradGuard no_grad;
    run_gpu_loop(state, bytes,
        [&]
        {
            output = conv->forward(input);
            torch::cuda::synchronize();
        });
}

// BM_Conv2d_Bwd_Custom / BM_Conv2d_Bwd_Torch: gradient of the same 64→192 convolution; Custom measures backward alone, LibTorch also zeroes the gradient.
static void BM_Conv2d_Bwd_Custom(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    // Conv shape: batch 16, 64→192, kernel 3, spatial 112.
    Conv2d conv(kConvIn, kConvOut, kKernel, 1, 1);
    conv.to(dl::Device::GPU);
    const std::vector<int> in_shape { kBatch, kConvIn, kSpatial, kSpatial };
    const std::vector<int> out_shape { kBatch, kConvOut, kSpatial, kSpatial };
    const dl::Tensor input = dl::Tensor::from_host(in_shape, host_filled(in_shape, 0.1F), dl::Device::GPU);
    dl::Tensor output = warmup_custom_conv(conv, input);
    const dl::Tensor grad_out = dl::Tensor::from_host(out_shape, host_filled(out_shape, 1.0F), dl::Device::GPU);
    dl::Tensor grad_in = conv.backward(grad_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    const std::size_t bytes = numel(out_shape) * sizeof(float);

    Profiler profiler;
    for (auto _ : state)
    {
        output = conv.forward(input);
        CHECK_CUDA(cudaDeviceSynchronize());
        profiler.start();
        grad_in = conv.backward(grad_out);
        const float milliseconds = profiler.stop();
        // Iteration time is the Profiler GPU interval, not the Google Benchmark wall clock.
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(bytes));
    state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
}

static void BM_Conv2d_Bwd_Torch(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    at::globalContext().setBenchmarkCuDNN(true);
    // Conv shape: batch 16, 64→192, kernel 3, spatial 112.
    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(kConvIn, kConvOut, kKernel).padding(1));
    conv->to(torch::kCUDA);
    conv->train();
    auto input = torch::full({ kBatch, kConvIn, kSpatial, kSpatial }, 0.1F,
        torch::device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true));
    auto grad_out = torch::ones({ kBatch, kConvOut, kSpatial, kSpatial }, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    for (int index = 0; index < kWarmup; ++index)
    {
        auto output = conv->forward(input);
        output.backward(grad_out);
        conv->zero_grad();
        if (input.grad().defined())
        {
            input.grad().zero_();
        }
    }
    torch::cuda::synchronize();
    auto output = conv->forward(input);
    torch::cuda::synchronize();
    const std::size_t bytes = static_cast<std::size_t>(grad_out.numel()) * sizeof(float);

    run_gpu_loop(state, bytes,
        [&]
        {
            output.backward(grad_out, /*retain_graph=*/true);
            conv->zero_grad();
            if (input.grad().defined())
            {
                input.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

// BM_FC_Fwd_Custom / BM_FC_Fwd_Torch: layer 7*7*1024 → 4096, input [16, 50176] (first linear layer of the head).
static void BM_FC_Fwd_Custom(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    // Head GEMM shape: 50176 x 16 x 4096 (7*7*1024, batch, 4096).
    FullyConnected fc(kFcIn, kFcOut);
    fc.to(dl::Device::GPU);
    const std::vector<int> shape { kBatch, kFcIn };
    const dl::Tensor input = dl::Tensor::from_host(shape, host_filled(shape, 0.05F), dl::Device::GPU);
    dl::Tensor output;
    for (int index = 0; index < kWarmup; ++index)
    {
        output = fc.forward(input);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    const std::size_t bytes = numel(shape) * sizeof(float);

    run_gpu_loop(state, bytes,
        [&]
        {
            output = fc.forward(input);
        });
}

static void BM_FC_Fwd_Torch(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    // Head GEMM shape: 50176 x 16 x 4096 (7*7*1024, batch, 4096).
    torch::nn::Linear fc(kFcIn, kFcOut);
    fc->to(torch::kCUDA);
    fc->eval();
    auto input = torch::full({ kBatch, kFcIn }, 0.05F, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        for (int index = 0; index < kWarmup; ++index)
        {
            output = fc->forward(input);
        }
        torch::cuda::synchronize();
    }
    const std::size_t bytes = static_cast<std::size_t>(input.numel()) * sizeof(float);

    torch::NoGradGuard no_grad;
    run_gpu_loop(state, bytes,
        [&]
        {
            output = fc->forward(input);
            torch::cuda::synchronize();
        });
}

// BM_FC_Bwd_Custom / BM_FC_Bwd_Torch: gradient of the 50176→4096 layer; Custom measures backward alone, with that iteration's forward outside the timer.
static void BM_FC_Bwd_Custom(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    // Head GEMM shape: 50176 x 16 x 4096 (7*7*1024, batch, 4096).
    FullyConnected fc(kFcIn, kFcOut);
    fc.to(dl::Device::GPU);
    const std::vector<int> in_shape { kBatch, kFcIn };
    const std::vector<int> out_shape { kBatch, kFcOut };
    const dl::Tensor input = dl::Tensor::from_host(in_shape, host_filled(in_shape, 0.05F), dl::Device::GPU);
    const dl::Tensor grad_out = dl::Tensor::from_host(out_shape, host_filled(out_shape, 1.0F), dl::Device::GPU);
    dl::Tensor grad_in;
    for (int index = 0; index < kWarmup; ++index)
    {
        (void)fc.forward(input);
        grad_in = fc.backward(grad_out);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    const std::size_t bytes = numel(out_shape) * sizeof(float);

    Profiler profiler;
    for (auto _ : state)
    {
        (void)fc.forward(input);
        CHECK_CUDA(cudaDeviceSynchronize());
        profiler.start();
        grad_in = fc.backward(grad_out);
        const float milliseconds = profiler.stop();
        // Iteration time is the Profiler GPU interval, not the Google Benchmark wall clock.
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(bytes));
    state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
}

static void BM_FC_Bwd_Torch(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    // Head GEMM shape: 50176 x 16 x 4096 (7*7*1024, batch, 4096).
    torch::nn::Linear fc(kFcIn, kFcOut);
    fc->to(torch::kCUDA);
    fc->train();
    auto input = torch::full(
        { kBatch, kFcIn }, 0.05F, torch::device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true));
    auto grad_out = torch::ones({ kBatch, kFcOut }, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    for (int index = 0; index < kWarmup; ++index)
    {
        auto output = fc->forward(input);
        output.backward(grad_out);
        fc->zero_grad();
        if (input.grad().defined())
        {
            input.grad().zero_();
        }
    }
    torch::cuda::synchronize();
    auto output = fc->forward(input);
    torch::cuda::synchronize();
    const std::size_t bytes = static_cast<std::size_t>(grad_out.numel()) * sizeof(float);

    run_gpu_loop(state, bytes,
        [&]
        {
            output.backward(grad_out, /*retain_graph=*/true);
            fc->zero_grad();
            if (input.grad().defined())
            {
                input.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

// BM_YOLOLoss_Fwd_Custom: YOLOv1 loss on the tensor [16, 7, 7, 30] with one object in cell (3, 3) and class 0.
static void BM_YOLOLoss_Fwd_Custom(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kBatch, kGrid, kGrid, kYoloDepth };
    const dl::Tensor prediction = dl::Tensor::from_host(shape, host_filled(shape, 0.2F), dl::Device::GPU);
    std::vector<float> target_host = host_filled(shape, 0.0F);
    for (int batch = 0; batch < kBatch; ++batch)
    {
        const std::size_t base = (static_cast<std::size_t>(batch) * static_cast<std::size_t>(kGrid) * kGrid * kYoloDepth)
            + (3ULL * kGrid * kYoloDepth) + (3ULL * kYoloDepth);
        target_host[base + 0] = 0.5F;
        target_host[base + 1] = 0.5F;
        target_host[base + 2] = 0.2F;
        target_host[base + 3] = 0.2F;
        target_host[base + 4] = 1.0F;
        target_host[base + 10] = 1.0F;
    }
    const dl::Tensor target = dl::Tensor::from_host(shape, target_host, dl::Device::GPU);
    dl::Tensor loss;
    for (int index = 0; index < kWarmup; ++index)
    {
        loss = YOLOLoss::loss(target, prediction, kYoloClasses);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    run_gpu_loop(state, numel(shape) * sizeof(float),
        [&]
        {
            loss = YOLOLoss::loss(target, prediction, kYoloClasses);
        });
}

// BM_YOLOLoss_Bwd_Custom: derivative of the same YOLOv1 loss with respect to the prediction [16, 7, 7, 30].
static void BM_YOLOLoss_Bwd_Custom(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kBatch, kGrid, kGrid, kYoloDepth };
    const dl::Tensor prediction = dl::Tensor::from_host(shape, host_filled(shape, 0.2F), dl::Device::GPU);
    std::vector<float> target_host = host_filled(shape, 0.0F);
    for (int batch = 0; batch < kBatch; ++batch)
    {
        const std::size_t base = (static_cast<std::size_t>(batch) * static_cast<std::size_t>(kGrid) * kGrid * kYoloDepth)
            + (3ULL * kGrid * kYoloDepth) + (3ULL * kYoloDepth);
        target_host[base + 0] = 0.5F;
        target_host[base + 1] = 0.5F;
        target_host[base + 2] = 0.2F;
        target_host[base + 3] = 0.2F;
        target_host[base + 4] = 1.0F;
        target_host[base + 10] = 1.0F;
    }
    const dl::Tensor target = dl::Tensor::from_host(shape, target_host, dl::Device::GPU);
    dl::Tensor grad;
    for (int index = 0; index < kWarmup; ++index)
    {
        grad = YOLOLoss::loss_derivative(target, prediction, kYoloClasses);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    run_gpu_loop(state, numel(shape) * sizeof(float),
        [&]
        {
            grad = YOLOLoss::loss_derivative(target, prediction, kYoloClasses);
        });
}

// LibTorch prediction and target with an object in cell (3, 3), the same layout as the Custom variant.
static auto make_torch_yolo_tensors() -> std::pair<torch::Tensor, torch::Tensor>
{
    auto prediction = torch::full({ kBatch, kGrid, kGrid, kYoloDepth }, 0.2F,
        torch::device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true));
    auto target = torch::zeros({ kBatch, kGrid, kGrid, kYoloDepth }, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    target.index_put_({ torch::indexing::Slice(), 3, 3, 0 }, 0.5);
    target.index_put_({ torch::indexing::Slice(), 3, 3, 1 }, 0.5);
    target.index_put_({ torch::indexing::Slice(), 3, 3, 2 }, 0.2);
    target.index_put_({ torch::indexing::Slice(), 3, 3, 3 }, 0.2);
    target.index_put_({ torch::indexing::Slice(), 3, 3, 4 }, 1.0);
    target.index_put_({ torch::indexing::Slice(), 3, 3, 10 }, 1.0);
    return { prediction, target };
}

// BM_YOLOLoss_Fwd_Torch: YOLOv1 loss in LibTorch for the grid [16, 7, 7, 30] with an object in cell (3, 3).
static void BM_YOLOLoss_Fwd_Torch(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    auto tensors = make_torch_yolo_tensors();
    auto& prediction = tensors.first;
    auto& target = tensors.second;
    torch::Tensor loss;
    {
        torch::NoGradGuard no_grad;
        for (int index = 0; index < kWarmup; ++index)
        {
            loss = compute_yolo_loss(prediction, target);
        }
        torch::cuda::synchronize();
    }

    torch::NoGradGuard no_grad;
    run_gpu_loop(state, static_cast<std::size_t>(prediction.numel()) * sizeof(float),
        [&]
        {
            loss = compute_yolo_loss(prediction, target);
            torch::cuda::synchronize();
        });
}

// BM_YOLOLoss_Bwd_Torch: YOLOv1 loss backward in LibTorch, with the graph retained and the prediction gradient zeroed.
static void BM_YOLOLoss_Bwd_Torch(benchmark::State& state)
{
    if (!require_cuda(state))
    {
        return;
    }
    auto tensors = make_torch_yolo_tensors();
    auto& prediction = tensors.first;
    auto& target = tensors.second;
    torch::Tensor loss = compute_yolo_loss(prediction, target);
    const auto ones = torch::ones_like(loss);
    for (int index = 0; index < kWarmup; ++index)
    {
        loss.backward(ones, /*retain_graph=*/true);
        if (prediction.grad().defined())
        {
            prediction.grad().zero_();
        }
    }
    torch::cuda::synchronize();

    run_gpu_loop(state, static_cast<std::size_t>(prediction.numel()) * sizeof(float),
        [&]
        {
            loss.backward(ones, /*retain_graph=*/true);
            if (prediction.grad().defined())
            {
                prediction.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

BENCHMARK(BM_H2D_Custom_FromHost)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_H2D_Torch_To)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_H2D_Custom_ReuseMemcpy)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_H2D_Torch_CopyInto)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_D2H_Custom_ToHost)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_D2H_Torch_Cpu)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Conv2d_Fwd_Custom)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Conv2d_Fwd_Torch)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Conv2d_Bwd_Custom)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Conv2d_Bwd_Torch)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FC_Fwd_Custom)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FC_Fwd_Torch)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FC_Bwd_Custom)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FC_Bwd_Torch)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_YOLOLoss_Fwd_Custom)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_YOLOLoss_Fwd_Torch)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_YOLOLoss_Bwd_Custom)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_YOLOLoss_Bwd_Torch)->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
