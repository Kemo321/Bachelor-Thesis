#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>

#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/YOLO.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/dataset.hpp"

static void BM_CustomYOLO_ManualTraining(benchmark::State& state)
{
    const int batch_size = static_cast<int>(state.range(0));
    const std::string data_root = "../../data/VOCdevkit";
    const float learning_rate = 1e-4F;

    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root + "/VOC2012", train_paths, val_paths, test_paths);

    if (train_paths.images.empty())
    {
        state.SkipWithError("No data in JPEGImages/Annotations!");
        return;
    }

    CustomDataLoader train_loader(train_paths, batch_size, false);

    YOLO custom_model;
    for (auto& layer : custom_model.get_all_layers()) {
        layer->to(dl::Device::GPU);
    }

    Network trainer(custom_model.get_all_layers(), learning_rate);
    int64_t total_processed = 0;

    for (auto _ : state)
    {
        train_loader.reset();
        while (train_loader.has_next())
        {
            Batch batch = train_loader.get_batch();

            dl::Tensor pred = custom_model.forward(batch.images);

            (void)YOLOLoss::loss(batch.targets, pred).to_host();
            dl::Tensor grad_error = YOLOLoss::loss_derivative(batch.targets, pred);
            grad_error = grad_error.clamp(-5.0F, 5.0F);

            auto layers = custom_model.get_all_layers();
            for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
            {
                grad_error = (*iterator)->backward(grad_error);
            }

            for (auto& layer : layers)
            {
                layer->step();
            }

            cudaDeviceSynchronize();
            total_processed += batch.images.get_shape()[0];
        }
    }

    state.SetItemsProcessed(total_processed);
    state.counters["Img/Sec"] = benchmark::Counter(
        static_cast<double>(total_processed),
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_CustomYOLO_ManualTraining)->Arg(8)->Arg(16)->UseRealTime();
BENCHMARK_MAIN();
