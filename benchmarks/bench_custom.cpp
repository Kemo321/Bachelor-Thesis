#include <benchmark/benchmark.h>
#include <iostream>
#include <torch/torch.h>

#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/dataset.hpp"

static void BM_CustomYOLO_ManualTraining(benchmark::State& state)
{
    const int batch_size = state.range(0);
    const std::string data_root = "../data/VOCdevkit";
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
        VOCYoloDataset(train_paths).map(torch::data::transforms::Stack<>()),
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

            float loss_val = YOLOLoss::loss(target, pred).item<float>();
            auto grad_error = YOLOLoss::loss_derivative(target, pred);

            grad_error = grad_error.clamp(-1.0, 1.0);

            auto layers = custom_model->get_all_layers();
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {
                grad_error = (*it)->backward(grad_error);
            }

            if (device.is_cuda()) { torch::cuda::synchronize(); }
            total_processed += data.size(0);
        }
    }

    state.SetItemsProcessed(total_processed);
    state.counters["Img/Sec"] = benchmark::Counter(static_cast<double>(total_processed), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_CustomYOLO_ManualTraining)->Arg(8)->Arg(16)->Iterations(1)->Unit(benchmark::kSecond)->UseRealTime();
BENCHMARK_MAIN();