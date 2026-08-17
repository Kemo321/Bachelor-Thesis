#include "bench_micro_common.hpp"

#include "DeepLearnLib/BatchNorm2d.hpp"
#include "DeepLearnLib/CSVLoader.hpp"
#include "DeepLearnLib/ClassificationLoader.hpp"
#include "DeepLearnLib/Dropout.hpp"
#include "DeepLearnLib/Flatten.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/FusedCBR2d.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/SimpleCNN.hpp"
#include "DeepLearnLib/Softmax.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/mAP.hpp"
#include "DeepLearnLib/utils.hpp"
#include "TorchDataset.hpp"
#include "TorchYOLO.hpp"

#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

auto repo_root() -> fs::path
{
#ifdef DEEPLEARN_SOURCE_DIR
    return fs::path(DEEPLEARN_SOURCE_DIR);
#else
    return fs::current_path().parent_path().parent_path();
#endif
}

auto bytes_of(const dl::Tensor& tensor) -> std::size_t
{
    return micro_numel(tensor.get_shape()) * sizeof(float);
}

struct VocSplit
{
    DataPaths train;
    bool ready { false };
};

auto voc_split() -> VocSplit&
{
    static VocSplit cache;
    static bool attempted { false };
    if (!attempted)
    {
        attempted = true;
        const fs::path voc = repo_root() / "data" / "VOCdevkit" / "VOC2012";
        if (fs::is_directory(voc))
        {
            DataPaths val;
            DataPaths test;
            split_dataset(voc.string(), cache.train, val, test);
            cache.ready = !cache.train.images.empty();
        }
    }
    return cache;
}

auto cifar_loader() -> ClassificationLoader*
{
    static std::unique_ptr<ClassificationLoader> loader;
    static bool attempted { false };
    if (!attempted)
    {
        attempted = true;
        const fs::path root = repo_root() / "data" / "cifar10";
        if (fs::is_directory(root / "train"))
        {
            try
            {
                loader = std::make_unique<ClassificationLoader>(root.string(), "train", kMicroBatch, 32, false);
            }
            catch (const std::exception&)
            {
                loader.reset();
            }
        }
    }
    return loader.get();
}

auto voc_loader() -> CustomDataLoader*
{
    static std::unique_ptr<CustomDataLoader> loader;
    static bool attempted { false };
    if (!attempted)
    {
        attempted = true;
        auto& voc = voc_split();
        if (voc.ready)
        {
            loader = std::make_unique<CustomDataLoader>(voc.train, kMicroBatch, false);
        }
    }
    return loader.get();
}

} // namespace

static void BM_Tensor_Add_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kMicroBatch, 512, 28, 28 };
    const dl::Tensor left = micro_gpu_tensor(shape, 0.3F);
    const dl::Tensor right = micro_gpu_tensor(shape, 0.7F);
    dl::Tensor out = left + right;
    micro_gpu_loop(state, bytes_of(left),
        [&]
        {
            out = left + right;
        });
}

static void BM_Tensor_Add_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto left = micro_torch_cuda({ kMicroBatch, 512, 28, 28 }, 0.3F);
    auto right = micro_torch_cuda({ kMicroBatch, 512, 28, 28 }, 0.7F);
    torch::Tensor out = left + right;
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(left.numel()) * sizeof(float),
        [&]
        {
            out = left + right;
            torch::cuda::synchronize();
        });
}

static void BM_Tensor_Mul_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kMicroBatch, 512, 28, 28 };
    const dl::Tensor left = micro_gpu_tensor(shape, 0.3F);
    const dl::Tensor right = micro_gpu_tensor(shape, 0.7F);
    dl::Tensor out = left * right;
    micro_gpu_loop(state, bytes_of(left),
        [&]
        {
            out = left * right;
        });
}

static void BM_Tensor_Mul_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto left = micro_torch_cuda({ kMicroBatch, 512, 28, 28 }, 0.3F);
    auto right = micro_torch_cuda({ kMicroBatch, 512, 28, 28 }, 0.7F);
    torch::Tensor out = left * right;
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(left.numel()) * sizeof(float),
        [&]
        {
            out = left * right;
            torch::cuda::synchronize();
        });
}

static void BM_Tensor_Matmul_FCGrad_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const dl::Tensor left = micro_gpu_tensor({ 7 * 7 * 1024, kMicroBatch }, 0.05F);
    const dl::Tensor right = micro_gpu_tensor({ kMicroBatch, 4096 }, 0.05F);
    dl::Tensor out = left.matmul(right);
    CHECK_CUDA(cudaDeviceSynchronize());
    micro_gpu_loop(state, bytes_of(left) + bytes_of(right),
        [&]
        {
            out = left.matmul(right);
        });
}

static void BM_Tensor_Matmul_FCGrad_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto left = micro_torch_cuda({ 7 * 7 * 1024, kMicroBatch }, 0.05F);
    auto right = micro_torch_cuda({ kMicroBatch, 4096 }, 0.05F);
    torch::Tensor out = torch::matmul(left, right);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(left.numel() + right.numel()) * sizeof(float),
        [&]
        {
            out = torch::matmul(left, right);
            torch::cuda::synchronize();
        });
}

static void BM_Tensor_Matmul_Head_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const dl::Tensor left = micro_gpu_tensor({ kMicroBatch, 4096 }, 0.05F);
    const dl::Tensor right = micro_gpu_tensor({ 4096, 1470 }, 0.05F);
    dl::Tensor out = left.matmul(right);
    CHECK_CUDA(cudaDeviceSynchronize());
    micro_gpu_loop(state, bytes_of(left) + bytes_of(right),
        [&]
        {
            out = left.matmul(right);
        });
}

static void BM_Tensor_Matmul_Head_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto left = micro_torch_cuda({ kMicroBatch, 4096 }, 0.05F);
    auto right = micro_torch_cuda({ 4096, 1470 }, 0.05F);
    torch::Tensor out = torch::matmul(left, right);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(left.numel() + right.numel()) * sizeof(float),
        [&]
        {
            out = torch::matmul(left, right);
            torch::cuda::synchronize();
        });
}

static void BM_Tensor_Transpose_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const dl::Tensor input = micro_gpu_tensor({ kMicroBatch, 7 * 7 * 1024 }, 0.05F);
    dl::Tensor out = input.transpose();
    CHECK_CUDA(cudaDeviceSynchronize());
    micro_gpu_loop(state, bytes_of(input),
        [&]
        {
            out = input.transpose();
        });
}

static void BM_Tensor_Transpose_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto input = micro_torch_cuda({ kMicroBatch, 7 * 7 * 1024 }, 0.05F);
    torch::Tensor out = input.transpose(0, 1).contiguous();
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            out = input.transpose(0, 1).contiguous();
            torch::cuda::synchronize();
        });
}

static void BM_Tensor_Sum_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const dl::Tensor input = micro_gpu_tensor({ kMicroBatch, 512, 28, 28 }, 0.1F);
    dl::Tensor out = input.sum();
    micro_gpu_loop(state, bytes_of(input),
        [&]
        {
            out = input.sum();
        });
}

static void BM_Tensor_Sum_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto input = micro_torch_cuda({ kMicroBatch, 512, 28, 28 }, 0.1F);
    torch::Tensor out = input.sum();
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            out = input.sum();
            torch::cuda::synchronize();
        });
}

static void BM_Tensor_Clamp_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const dl::Tensor input = micro_gpu_tensor({ kMicroBatch, 1024, 7, 7 }, 0.1F);
    dl::Tensor out = input.clamp(-1.0F, 1.0F);
    micro_gpu_loop(state, bytes_of(input),
        [&]
        {
            out = input.clamp(-1.0F, 1.0F);
        });
}

static void BM_Tensor_Clamp_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto input = micro_torch_cuda({ kMicroBatch, 1024, 7, 7 }, 0.1F);
    torch::Tensor out = input.clamp(-1.0, 1.0);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            out = input.clamp(-1.0, 1.0);
            torch::cuda::synchronize();
        });
}

static void BM_Tensor_ToFp16_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const dl::Tensor input = micro_gpu_tensor({ kMicroBatch, kMicroChannels, kMicroImage, kMicroImage }, 0.1F);
    dl::Tensor out = input.to_dtype(dl::Dtype::Float16);
    CHECK_CUDA(cudaDeviceSynchronize());
    micro_gpu_loop(state, bytes_of(input),
        [&]
        {
            out = input.to_dtype(dl::Dtype::Float16);
        });
}

static void BM_Tensor_ToFp16_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto input = micro_torch_cuda({ kMicroBatch, kMicroChannels, kMicroImage, kMicroImage }, 0.1F);
    torch::Tensor out = input.to(torch::kFloat16);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            out = input.to(torch::kFloat16);
            torch::cuda::synchronize();
        });
}

static void BM_BN_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    BatchNorm2d layer(192);
    layer.train();
    micro_custom_fwd(state, layer, micro_gpu_tensor({ kMicroBatch, 192, 112, 112 }, 0.1F));
}

static void BM_BN_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::BatchNorm2d layer(192);
    layer->to(torch::kCUDA);
    layer->train();
    auto input = micro_torch_cuda({ kMicroBatch, 192, 112, 112 }, 0.1F);
    torch::Tensor output;
    torch::NoGradGuard no_grad;
    for (int index = 0; index < kMicroWarmup; ++index)
    {
        output = layer->forward(input);
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = layer->forward(input);
            torch::cuda::synchronize();
        });
}

static void BM_BN_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    BatchNorm2d layer(192);
    layer.train();
    micro_custom_bwd(state, layer, micro_gpu_tensor({ kMicroBatch, 192, 112, 112 }, 0.1F));
}

static void BM_BN_Bwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::BatchNorm2d layer(192);
    layer->to(torch::kCUDA);
    layer->train();
    auto input = micro_torch_cuda({ kMicroBatch, 192, 112, 112 }, 0.1F, true);
    auto output = layer->forward(input);
    auto grad = torch::ones_like(output);
    output.backward(grad, true);
    layer->zero_grad();
    if (input.grad().defined())
    {
        input.grad().zero_();
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(output.numel()) * sizeof(float),
        [&]
        {
            output.backward(grad, true);
            layer->zero_grad();
            if (input.grad().defined())
            {
                input.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

static void BM_MaxPool_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    MaxPool2d layer(2, 2);
    micro_custom_fwd(state, layer, micro_gpu_tensor({ kMicroBatch, 192, 112, 112 }, 0.1F));
}

static void BM_MaxPool_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::MaxPool2d layer(torch::nn::MaxPool2dOptions(2).stride(2));
    layer->to(torch::kCUDA);
    auto input = micro_torch_cuda({ kMicroBatch, 192, 112, 112 }, 0.1F);
    torch::Tensor output;
    torch::NoGradGuard no_grad;
    for (int index = 0; index < kMicroWarmup; ++index)
    {
        output = layer->forward(input);
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = layer->forward(input);
            torch::cuda::synchronize();
        });
}

static void BM_MaxPool_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    MaxPool2d layer(2, 2);
    micro_custom_bwd(state, layer, micro_gpu_tensor({ kMicroBatch, 192, 112, 112 }, 0.1F));
}

static void BM_MaxPool_Bwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::MaxPool2d layer(torch::nn::MaxPool2dOptions(2).stride(2));
    layer->to(torch::kCUDA);
    auto input = micro_torch_cuda({ kMicroBatch, 192, 112, 112 }, 0.1F, true);
    auto output = layer->forward(input);
    auto grad = torch::ones_like(output);
    output.backward(grad, true);
    if (input.grad().defined())
    {
        input.grad().zero_();
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(output.numel()) * sizeof(float),
        [&]
        {
            output.backward(grad, true);
            if (input.grad().defined())
            {
                input.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

static void BM_LeakyReLU_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    LeakyReLU layer(0.1F);
    micro_custom_fwd(state, layer, micro_gpu_tensor({ kMicroBatch, 192, 112, 112 }, 0.1F));
}

static void BM_LeakyReLU_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::LeakyReLU layer(torch::nn::LeakyReLUOptions().negative_slope(0.1));
    layer->to(torch::kCUDA);
    auto input = micro_torch_cuda({ kMicroBatch, 192, 112, 112 }, 0.1F);
    torch::Tensor output;
    torch::NoGradGuard no_grad;
    output = layer->forward(input);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = layer->forward(input);
            torch::cuda::synchronize();
        });
}

static void BM_LeakyReLU_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    LeakyReLU layer(0.1F);
    micro_custom_bwd(state, layer, micro_gpu_tensor({ kMicroBatch, 192, 112, 112 }, 0.1F));
}

static void BM_LeakyReLU_Bwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::LeakyReLU layer(torch::nn::LeakyReLUOptions().negative_slope(0.1));
    layer->to(torch::kCUDA);
    auto input = micro_torch_cuda({ kMicroBatch, 192, 112, 112 }, 0.1F, true);
    auto output = layer->forward(input);
    auto grad = torch::ones_like(output);
    output.backward(grad, true);
    if (input.grad().defined())
    {
        input.grad().zero_();
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(output.numel()) * sizeof(float),
        [&]
        {
            output.backward(grad, true);
            if (input.grad().defined())
            {
                input.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

static void BM_Dropout_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    Dropout layer(0.5F);
    layer.train();
    micro_custom_fwd(state, layer, micro_gpu_tensor({ kMicroBatch, 4096 }, 0.1F));
}

static void BM_Dropout_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::Dropout layer(torch::nn::DropoutOptions(0.5));
    layer->to(torch::kCUDA);
    layer->train();
    auto input = micro_torch_cuda({ kMicroBatch, 4096 }, 0.1F);
    torch::Tensor output;
    output = layer->forward(input);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = layer->forward(input);
            torch::cuda::synchronize();
        });
}

static void BM_Dropout_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    Dropout layer(0.5F);
    layer.train();
    micro_custom_bwd(state, layer, micro_gpu_tensor({ kMicroBatch, 4096 }, 0.1F));
}

static void BM_Dropout_Bwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::Dropout layer(torch::nn::DropoutOptions(0.5));
    layer->to(torch::kCUDA);
    layer->train();
    auto input = micro_torch_cuda({ kMicroBatch, 4096 }, 0.1F, true);
    auto output = layer->forward(input);
    auto grad = torch::ones_like(output);
    output.backward(grad, true);
    if (input.grad().defined())
    {
        input.grad().zero_();
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(output.numel()) * sizeof(float),
        [&]
        {
            output.backward(grad, true);
            if (input.grad().defined())
            {
                input.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

static void BM_Flatten_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    Flatten layer;
    micro_custom_fwd(state, layer, micro_gpu_tensor({ kMicroBatch, 1024, 7, 7 }, 0.1F));
}

static void BM_Flatten_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto input = micro_torch_cuda({ kMicroBatch, 1024, 7, 7 }, 0.1F);
    torch::Tensor output = input.view({ kMicroBatch, -1 });
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = input.view({ kMicroBatch, -1 });
            torch::cuda::synchronize();
        });
}

static void BM_Flatten_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    Flatten layer;
    micro_custom_bwd(state, layer, micro_gpu_tensor({ kMicroBatch, 1024, 7, 7 }, 0.1F));
}

static void BM_Softmax_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    Softmax layer;
    micro_custom_fwd(state, layer, micro_gpu_tensor({ kMicroBatch, 10 }, 0.1F));
}

static void BM_Softmax_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto input = micro_torch_cuda({ kMicroBatch, 10 }, 0.1F);
    torch::Tensor output = torch::softmax(input, 1);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = torch::softmax(input, 1);
            torch::cuda::synchronize();
        });
}

static void BM_Softmax_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    Softmax layer;
    micro_custom_bwd(state, layer, micro_gpu_tensor({ kMicroBatch, 10 }, 0.1F));
}

static void BM_Softmax_Bwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto input = micro_torch_cuda({ kMicroBatch, 10 }, 0.1F, true);
    auto output = torch::softmax(input, 1);
    auto grad = torch::ones_like(output);
    output.backward(grad, true);
    if (input.grad().defined())
    {
        input.grad().zero_();
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(output.numel()) * sizeof(float),
        [&]
        {
            output.backward(grad, true);
            if (input.grad().defined())
            {
                input.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

static void BM_FusedCBR_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    FusedCBR2d layer(64, 192, 3, 1, 1, 0.1F);
    layer.train();
    micro_custom_fwd(state, layer, micro_gpu_tensor({ kMicroBatch, 64, 112, 112 }, 0.1F));
}

static void BM_FusedCBR_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    at::globalContext().setBenchmarkCuDNN(true);
    torch::nn::Sequential layer(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 3).padding(1)),
        torch::nn::BatchNorm2d(192), torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
    layer->to(torch::kCUDA);
    layer->train();
    auto input = micro_torch_cuda({ kMicroBatch, 64, 112, 112 }, 0.1F);
    torch::Tensor output;
    torch::NoGradGuard no_grad;
    for (int index = 0; index < kMicroWarmup; ++index)
    {
        output = layer->forward(input);
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = layer->forward(input);
            torch::cuda::synchronize();
        });
}

static void BM_FusedCBR_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    FusedCBR2d layer(64, 192, 3, 1, 1, 0.1F);
    layer.train();
    micro_custom_bwd(state, layer, micro_gpu_tensor({ kMicroBatch, 64, 112, 112 }, 0.1F));
}

static void BM_FusedCBR_Bwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    at::globalContext().setBenchmarkCuDNN(true);
    torch::nn::Sequential layer(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 3).padding(1)),
        torch::nn::BatchNorm2d(192), torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
    layer->to(torch::kCUDA);
    layer->train();
    auto input = micro_torch_cuda({ kMicroBatch, 64, 112, 112 }, 0.1F, true);
    auto output = layer->forward(input);
    auto grad = torch::ones_like(output);
    output.backward(grad, true);
    layer->zero_grad();
    if (input.grad().defined())
    {
        input.grad().zero_();
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(output.numel()) * sizeof(float),
        [&]
        {
            output.backward(grad, true);
            layer->zero_grad();
            if (input.grad().defined())
            {
                input.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

static void BM_FusedCBR_Stem_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    FusedCBR2d layer(3, 64, 7, 2, 3, 0.1F);
    layer.train();
    micro_custom_fwd(state, layer, micro_gpu_tensor({ kMicroBatch, 3, kMicroImage, kMicroImage }, 0.1F));
}

static void BM_FusedCBR_Stem_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    at::globalContext().setBenchmarkCuDNN(true);
    torch::nn::Sequential layer(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)),
        torch::nn::BatchNorm2d(64), torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
    layer->to(torch::kCUDA);
    layer->train();
    auto input = micro_torch_cuda({ kMicroBatch, 3, kMicroImage, kMicroImage }, 0.1F);
    torch::Tensor output;
    torch::NoGradGuard no_grad;
    for (int index = 0; index < kMicroWarmup; ++index)
    {
        output = layer->forward(input);
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = layer->forward(input);
            torch::cuda::synchronize();
        });
}

static void BM_FC_Head2_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    FullyConnected layer(4096, 1470);
    micro_custom_fwd(state, layer, micro_gpu_tensor({ kMicroBatch, 4096 }, 0.05F));
}

static void BM_FC_Head2_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::Linear layer(4096, 1470);
    layer->to(torch::kCUDA);
    layer->eval();
    auto input = micro_torch_cuda({ kMicroBatch, 4096 }, 0.05F);
    torch::Tensor output;
    torch::NoGradGuard no_grad;
    output = layer->forward(input);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = layer->forward(input);
            torch::cuda::synchronize();
        });
}

static void BM_FC_Head2_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    FullyConnected layer(4096, 1470);
    micro_custom_bwd(state, layer, micro_gpu_tensor({ kMicroBatch, 4096 }, 0.05F));
}

static void BM_FC_Head2_Bwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::Linear layer(4096, 1470);
    layer->to(torch::kCUDA);
    layer->train();
    auto input = micro_torch_cuda({ kMicroBatch, 4096 }, 0.05F, true);
    auto output = layer->forward(input);
    auto grad = torch::ones_like(output);
    output.backward(grad, true);
    layer->zero_grad();
    if (input.grad().defined())
    {
        input.grad().zero_();
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(output.numel()) * sizeof(float),
        [&]
        {
            output.backward(grad, true);
            layer->zero_grad();
            if (input.grad().defined())
            {
                input.grad().zero_();
            }
            torch::cuda::synchronize();
        });
}

static void BM_MSE_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kMicroBatch, 10 };
    const dl::Tensor pred = micro_gpu_tensor(shape, 0.2F);
    const dl::Tensor target = micro_gpu_tensor(shape, 0.1F);
    dl::Tensor loss = MSELoss::loss(target, pred);
    micro_gpu_loop(state, bytes_of(pred),
        [&]
        {
            loss = MSELoss::loss(target, pred);
        });
}

static void BM_MSE_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto pred = micro_torch_cuda({ kMicroBatch, 10 }, 0.2F);
    auto target = micro_torch_cuda({ kMicroBatch, 10 }, 0.1F);
    torch::Tensor loss = torch::mse_loss(pred, target);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(pred.numel()) * sizeof(float),
        [&]
        {
            loss = torch::mse_loss(pred, target);
            torch::cuda::synchronize();
        });
}

static void BM_MSE_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kMicroBatch, 10 };
    const dl::Tensor pred = micro_gpu_tensor(shape, 0.2F);
    const dl::Tensor target = micro_gpu_tensor(shape, 0.1F);
    dl::Tensor grad = MSELoss::loss_derivative(target, pred);
    micro_gpu_loop(state, bytes_of(pred),
        [&]
        {
            grad = MSELoss::loss_derivative(target, pred);
        });
}

static void BM_CE_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kMicroBatch, 10 };
    const dl::Tensor pred = micro_gpu_tensor(shape, 0.2F);
    std::vector<float> target_host(micro_numel(shape), 0.0F);
    for (int batch = 0; batch < kMicroBatch; ++batch)
    {
        target_host[static_cast<std::size_t>(batch) * 10U + 3U] = 1.0F;
    }
    const dl::Tensor target = dl::Tensor::from_host(shape, target_host, dl::Device::GPU);
    dl::Tensor loss = CrossEntropyLoss::loss(target, pred);
    micro_gpu_loop(state, bytes_of(pred),
        [&]
        {
            loss = CrossEntropyLoss::loss(target, pred);
        });
}

static void BM_CE_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto pred = micro_torch_cuda({ kMicroBatch, 10 }, 0.2F);
    auto target = torch::full({ kMicroBatch }, 3, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    torch::Tensor loss = torch::nn::functional::cross_entropy(pred, target);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(pred.numel()) * sizeof(float),
        [&]
        {
            loss = torch::nn::functional::cross_entropy(pred, target);
            torch::cuda::synchronize();
        });
}

static void BM_CE_Bwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const std::vector<int> shape { kMicroBatch, 10 };
    const dl::Tensor pred = micro_gpu_tensor(shape, 0.2F);
    std::vector<float> target_host(micro_numel(shape), 0.0F);
    for (int batch = 0; batch < kMicroBatch; ++batch)
    {
        target_host[static_cast<std::size_t>(batch) * 10U + 3U] = 1.0F;
    }
    const dl::Tensor target = dl::Tensor::from_host(shape, target_host, dl::Device::GPU);
    dl::Tensor grad = CrossEntropyLoss::loss_derivative(target, pred);
    micro_gpu_loop(state, bytes_of(pred),
        [&]
        {
            grad = CrossEntropyLoss::loss_derivative(target, pred);
        });
}

static void BM_ClipGrad_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const dl::Tensor grad = micro_gpu_tensor({ kMicroBatch, 7, 7, 30 }, 5.0F);
    Network trainer({}, 1e-4F, 10.0F);
    dl::Tensor clipped = trainer.clip_loss_gradient(grad);
    micro_gpu_loop(state, bytes_of(grad),
        [&]
        {
            clipped = trainer.clip_loss_gradient(grad);
        });
}

static void BM_ClipGrad_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto grad = micro_torch_cuda({ kMicroBatch, 7, 7, 30 }, 5.0F);
    torch::Tensor clipped = torch::clamp(grad, -10.0, 10.0);
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(grad.numel()) * sizeof(float),
        [&]
        {
            clipped = torch::clamp(grad, -10.0, 10.0);
            torch::cuda::synchronize();
        });
}

static void BM_YOLO_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    YOLO model(20);
    for (auto& layer : model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->eval();
    }
    const dl::Tensor input = micro_gpu_tensor({ kMicroBatch, 3, kMicroImage, kMicroImage }, 0.1F);
    dl::Tensor output;
    for (int index = 0; index < kMicroModelWarmup; ++index)
    {
        output = model.forward(input);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    micro_gpu_loop(state, bytes_of(input),
        [&]
        {
            output = model.forward(input);
        });
}

static void BM_YOLO_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    at::globalContext().setBenchmarkCuDNN(true);
    YOLOv1 model(20);
    model->to(torch::kCUDA);
    model->eval();
    auto input = micro_torch_cuda({ kMicroBatch, 3, kMicroImage, kMicroImage }, 0.1F);
    torch::Tensor output;
    torch::NoGradGuard no_grad;
    for (int index = 0; index < kMicroModelWarmup; ++index)
    {
        output = model->forward(input);
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = model->forward(input);
            torch::cuda::synchronize();
        });
}

static void BM_YOLO_TrainStep_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    YOLO model(20);
    for (auto& layer : model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->train();
        layer->learning_rate = 1e-4F;
    }
    Network trainer(model.get_all_layers(), 1e-4F, 10.0F);
    const dl::Tensor input = micro_gpu_tensor({ kMicroBatch, 3, kMicroImage, kMicroImage }, 0.1F);
    const dl::Tensor target = micro_gpu_tensor({ kMicroBatch, 7, 7, 30 }, 0.0F);
    auto layers = model.get_all_layers();
    auto one_step = [&]()
    {
        dl::Tensor pred = model.forward(input);
        (void)YOLOLoss::loss(target, pred, 20);
        dl::Tensor grad = trainer.clip_loss_gradient(YOLOLoss::loss_derivative(target, pred, 20));
        for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
        {
            grad = (*iterator)->backward(grad);
        }
        for (auto& layer : layers)
        {
            layer->step();
        }
    };
    for (int index = 0; index < kMicroModelWarmup; ++index)
    {
        one_step();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    micro_gpu_loop(state, bytes_of(input),
        [&]
        {
            one_step();
        });
}

static void BM_YOLO_TrainStep_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    at::globalContext().setBenchmarkCuDNN(true);
    YOLOv1 model(20);
    model->to(torch::kCUDA);
    model->train();
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(1e-4));
    auto input = micro_torch_cuda({ kMicroBatch, 3, kMicroImage, kMicroImage }, 0.1F);
    auto target = micro_torch_cuda({ kMicroBatch, 7, 7, 30 }, 0.0F);
    auto one_step = [&]()
    {
        optimizer.zero_grad();
        auto pred = model->forward(input);
        auto loss = compute_yolo_loss(pred, target);
        loss.backward();
        optimizer.step();
        torch::cuda::synchronize();
    };
    for (int index = 0; index < kMicroModelWarmup; ++index)
    {
        one_step();
    }
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            one_step();
        });
}

static void BM_SimpleCNN_Fwd_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    SimpleCNN model(10, 32);
    for (auto& layer : model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->eval();
    }
    const dl::Tensor input = micro_gpu_tensor({ kMicroBatch, 3, 32, 32 }, 0.1F);
    dl::Tensor output;
    for (int index = 0; index < kMicroWarmup; ++index)
    {
        output = model.forward_logits(input);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    micro_gpu_loop(state, bytes_of(input),
        [&]
        {
            output = model.forward_logits(input);
        });
}

static void BM_SimpleCNN_Fwd_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    torch::nn::Sequential model(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),         torch::nn::Flatten(),
        torch::nn::Linear(32 * 8 * 8, 10));
    model->to(torch::kCUDA);
    model->eval();
    auto input = micro_torch_cuda({ kMicroBatch, 3, 32, 32 }, 0.1F);
    torch::Tensor output;
    torch::NoGradGuard no_grad;
    for (int index = 0; index < kMicroWarmup; ++index)
    {
        output = model->forward(input);
    }
    torch::cuda::synchronize();
    micro_gpu_loop(state, static_cast<std::size_t>(input.numel()) * sizeof(float),
        [&]
        {
            output = model->forward(input);
            torch::cuda::synchronize();
        });
}

static void BM_Loader_VOC_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    CustomDataLoader* loader = voc_loader();
    if (loader == nullptr)
    {
        state.SkipWithError("VOC dataset not found");
        return;
    }
    loader->reset();
    Batch batch = loader->get_batch();
    CHECK_CUDA(cudaDeviceSynchronize());
    const std::size_t bytes = bytes_of(batch.images);
    Profiler profiler;
    for (auto _ : state)
    {
        if (!loader->has_next())
        {
            loader->reset();
        }
        profiler.start();
        batch = loader->get_batch();
        const float milliseconds = profiler.stop();
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(bytes));
    state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
}

static void BM_Loader_VOC_Torch(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    auto& voc = voc_split();
    if (!voc.ready)
    {
        state.SkipWithError("VOC dataset not found");
        return;
    }
    VOCYoloDataset dataset(voc.train, false);
    const auto count = dataset.size().value_or(0);
    if (count == 0)
    {
        state.SkipWithError("VOC dataset empty");
        return;
    }
    std::size_t cursor = 0;
    auto load_batch = [&]()
    {
        std::vector<torch::Tensor> images;
        std::vector<torch::Tensor> targets;
        images.reserve(static_cast<std::size_t>(kMicroBatch));
        targets.reserve(static_cast<std::size_t>(kMicroBatch));
        for (int index = 0; index < kMicroBatch; ++index)
        {
            auto example = dataset.get(cursor % static_cast<std::size_t>(count));
            ++cursor;
            images.push_back(example.data);
            targets.push_back(example.target);
        }
        auto stacked = torch::stack(images).to(torch::kCUDA);
        auto stacked_t = torch::stack(targets).to(torch::kCUDA);
        torch::cuda::synchronize();
        return stacked.numel();
    };
    const auto elems = load_batch();
    Profiler profiler;
    for (auto _ : state)
    {
        profiler.start();
        (void)load_batch();
        const float milliseconds = profiler.stop();
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(elems) * static_cast<int64_t>(sizeof(float)));
    state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
}

static void BM_Loader_CIFAR_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    ClassificationLoader* loader = cifar_loader();
    if (loader == nullptr)
    {
        state.SkipWithError("CIFAR-10 dataset not found");
        return;
    }
    loader->reset();
    Batch batch = loader->get_batch();
    CHECK_CUDA(cudaDeviceSynchronize());
    const std::size_t bytes = bytes_of(batch.images);
    Profiler profiler;
    for (auto _ : state)
    {
        if (!loader->has_next())
        {
            loader->reset();
        }
        profiler.start();
        batch = loader->get_batch();
        const float milliseconds = profiler.stop();
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(bytes));
    state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
}

static void BM_Loader_CSV_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const fs::path csv = repo_root() / "data" / "tabular" / "demo.csv";
    if (!fs::is_regular_file(csv))
    {
        state.SkipWithError("tabular demo.csv not found");
        return;
    }
    Profiler profiler;
    for (auto _ : state)
    {
        profiler.start();
        CSVLoader loader(csv.string(), 1, true);
        CHECK_CUDA(cudaDeviceSynchronize());
        const float milliseconds = profiler.stop();
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
        state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
        (void)loader.size();
    }
}

static void BM_DecodeNMS_Custom(benchmark::State& state)
{
    if (!micro_require_cuda(state))
    {
        return;
    }
    const std::vector<float> output(static_cast<std::size_t>(7 * 7 * 30), 0.2F);
    std::vector<Detection> decoded = decode_yolo_tensor(output, 0.05F, 448, 448, 20);
    Profiler profiler;
    for (auto _ : state)
    {
        profiler.start();
        auto raw = decode_yolo_tensor(output, 0.05F, 448, 448, 20);
        auto kept = apply_nms(raw, 0.5F);
        const float milliseconds = profiler.stop();
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
        benchmark::DoNotOptimize(kept);
    }
    state.counters["VRAM_MiB"] = static_cast<double>(Profiler::get_vram_usage_mb());
}

static void BM_mAP_Custom(benchmark::State& state)
{
    std::vector<Detection> predicted;
    std::vector<Detection> truth;
    predicted.reserve(64);
    truth.reserve(32);
    for (int index = 0; index < 32; ++index)
    {
        truth.push_back(Detection { static_cast<float>(index) * 4.0F, 10.0F, 20.0F, 20.0F, 1.0F, index % 5 });
        predicted.push_back(
            Detection { static_cast<float>(index) * 4.0F + 1.0F, 11.0F, 20.0F, 20.0F, 0.9F, index % 5 });
        predicted.push_back(
            Detection { static_cast<float>(index) * 8.0F, 40.0F, 10.0F, 10.0F, 0.2F, (index + 1) % 5 });
    }
    Profiler profiler;
    for (auto _ : state)
    {
        profiler.start();
        float map = mean_average_precision(predicted, truth, 0.5F);
        const float milliseconds = profiler.stop();
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
        benchmark::DoNotOptimize(map);
    }
}

#define MICRO_BENCH(name) BENCHMARK(name)->UseManualTime()->Unit(benchmark::kMillisecond)

MICRO_BENCH(BM_Tensor_Add_Custom);
MICRO_BENCH(BM_Tensor_Add_Torch);
MICRO_BENCH(BM_Tensor_Mul_Custom);
MICRO_BENCH(BM_Tensor_Mul_Torch);
MICRO_BENCH(BM_Tensor_Matmul_FCGrad_Custom);
MICRO_BENCH(BM_Tensor_Matmul_FCGrad_Torch);
MICRO_BENCH(BM_Tensor_Matmul_Head_Custom);
MICRO_BENCH(BM_Tensor_Matmul_Head_Torch);
MICRO_BENCH(BM_Tensor_Transpose_Custom);
MICRO_BENCH(BM_Tensor_Transpose_Torch);
MICRO_BENCH(BM_Tensor_Sum_Custom);
MICRO_BENCH(BM_Tensor_Sum_Torch);
MICRO_BENCH(BM_Tensor_Clamp_Custom);
MICRO_BENCH(BM_Tensor_Clamp_Torch);
MICRO_BENCH(BM_Tensor_ToFp16_Custom);
MICRO_BENCH(BM_Tensor_ToFp16_Torch);
MICRO_BENCH(BM_BN_Fwd_Custom);
MICRO_BENCH(BM_BN_Fwd_Torch);
MICRO_BENCH(BM_BN_Bwd_Custom);
MICRO_BENCH(BM_BN_Bwd_Torch);
MICRO_BENCH(BM_MaxPool_Fwd_Custom);
MICRO_BENCH(BM_MaxPool_Fwd_Torch);
MICRO_BENCH(BM_MaxPool_Bwd_Custom);
MICRO_BENCH(BM_MaxPool_Bwd_Torch);
MICRO_BENCH(BM_LeakyReLU_Fwd_Custom);
MICRO_BENCH(BM_LeakyReLU_Fwd_Torch);
MICRO_BENCH(BM_LeakyReLU_Bwd_Custom);
MICRO_BENCH(BM_LeakyReLU_Bwd_Torch);
MICRO_BENCH(BM_Dropout_Fwd_Custom);
MICRO_BENCH(BM_Dropout_Fwd_Torch);
MICRO_BENCH(BM_Dropout_Bwd_Custom);
MICRO_BENCH(BM_Dropout_Bwd_Torch);
MICRO_BENCH(BM_Flatten_Fwd_Custom);
MICRO_BENCH(BM_Flatten_Fwd_Torch);
MICRO_BENCH(BM_Flatten_Bwd_Custom);
MICRO_BENCH(BM_Softmax_Fwd_Custom);
MICRO_BENCH(BM_Softmax_Fwd_Torch);
MICRO_BENCH(BM_Softmax_Bwd_Custom);
MICRO_BENCH(BM_Softmax_Bwd_Torch);
MICRO_BENCH(BM_FusedCBR_Fwd_Custom);
MICRO_BENCH(BM_FusedCBR_Fwd_Torch);
MICRO_BENCH(BM_FusedCBR_Bwd_Custom);
MICRO_BENCH(BM_FusedCBR_Bwd_Torch);
MICRO_BENCH(BM_FusedCBR_Stem_Fwd_Custom);
MICRO_BENCH(BM_FusedCBR_Stem_Fwd_Torch);
MICRO_BENCH(BM_FC_Head2_Fwd_Custom);
MICRO_BENCH(BM_FC_Head2_Fwd_Torch);
MICRO_BENCH(BM_FC_Head2_Bwd_Custom);
MICRO_BENCH(BM_FC_Head2_Bwd_Torch);
MICRO_BENCH(BM_MSE_Fwd_Custom);
MICRO_BENCH(BM_MSE_Fwd_Torch);
MICRO_BENCH(BM_MSE_Bwd_Custom);
MICRO_BENCH(BM_CE_Fwd_Custom);
MICRO_BENCH(BM_CE_Fwd_Torch);
MICRO_BENCH(BM_CE_Bwd_Custom);
MICRO_BENCH(BM_ClipGrad_Custom);
MICRO_BENCH(BM_ClipGrad_Torch);
MICRO_BENCH(BM_YOLO_Fwd_Custom);
MICRO_BENCH(BM_YOLO_Fwd_Torch);
MICRO_BENCH(BM_YOLO_TrainStep_Custom);
MICRO_BENCH(BM_YOLO_TrainStep_Torch);
MICRO_BENCH(BM_SimpleCNN_Fwd_Custom);
MICRO_BENCH(BM_SimpleCNN_Fwd_Torch);
MICRO_BENCH(BM_Loader_VOC_Custom);
MICRO_BENCH(BM_Loader_VOC_Torch);
MICRO_BENCH(BM_Loader_CIFAR_Custom);
MICRO_BENCH(BM_Loader_CSV_Custom);
MICRO_BENCH(BM_DecodeNMS_Custom);
MICRO_BENCH(BM_mAP_Custom);

#undef MICRO_BENCH
