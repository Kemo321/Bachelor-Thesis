#include <benchmark/benchmark.h>
#include <iostream>
#include <filesystem>
#include <torch/torch.h>

#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

static void BM_CustomYOLO_ManualTraining(benchmark::State& state)
{
    const int batch_size = state.range(0);
    // Ujednolicona sciezka wsadowa wychodzaca z build/benchmarks
    const std::string data_root = "../../data/VOCdevkit";
    const float learning_rate = 1e-4F;

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root + "/VOC2012", train_paths, val_paths, test_paths);

    if (train_paths.images.empty())
    {
        state.SkipWithError("Brak danych w JPEGImages/Annotations!");
        return;
    }

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths, false).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    YOLO custom_model;
    for (auto& layer : custom_model->get_all_layers()) {
        layer->to(device);
    }

    Network trainer(custom_model->get_all_layers(), learning_rate);
    int64_t total_processed = 0;

    for (auto _ : state)
    {
        torch::NoGradGuard no_grad;
        for (auto& batch : *train_loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            auto pred = custom_model->forward(data);

            // POPRAWKA: .item().toFloat() zamiast .item<float>()
            float loss_val = YOLOLoss::loss(target, pred).item().toFloat();
            auto grad_error = YOLOLoss::loss_derivative(target, pred);
            grad_error = grad_error.clamp(-5.0, 5.0);

            auto layers = custom_model->get_all_layers();
            for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
            {
                grad_error = (*iterator)->backward(grad_error);
            }

            for (auto& layer : layers)
            {
                layer->step();
            }

            if (device.is_cuda()) { torch::cuda::synchronize(); }
            total_processed += data.size(0);
        }
    }

    state.SetItemsProcessed(total_processed);
    state.counters["Img/Sec"] = benchmark::Counter(
        static_cast<double>(total_processed),
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_CustomYOLO_ManualTraining)->Arg(8)->Arg(16)->UseRealTime();
BENCHMARK_MAIN();