#include <benchmark/benchmark.h>
#include <iostream>
#include <limits>
#include <filesystem>
#include <torch/torch.h>

#include "DeepLearnLib/TorchYOLO.hpp"
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

namespace fs = std::filesystem;

static void BM_YOLOv1_SingleEpochTraining(benchmark::State& state)
{
    const int batch_size = state.range(0);
    // Ujednolicona sciezka wsadowa wychodzaca z build/benchmarks
    const std::string data_root = "../../data/VOCdevkit";
    const std::string results_dir = "../../results";

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

    YOLOv1 model;
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));

    int64_t total_processed = 0;

    for (auto _ : state)
    {
        float epoch_loss = 0.0F;
        model->train();

        for (auto& batch : *train_loader)
        {
            auto data = batch.data.to(device, true);
            auto target = batch.target.to(device, true);

            optimizer.zero_grad();
            auto pred = model->forward(data);
            auto loss = YOLOLoss::loss(target, pred);

            loss.backward();
            optimizer.step();

            if (device.is_cuda())
            {
                torch::cuda::synchronize();
            }

            // POPRAWKA: .item().toFloat() zamiast .item<float>()
            epoch_loss += loss.item().toFloat();
            total_processed += data.size(0);
        }

        benchmark::DoNotOptimize(epoch_loss);
    }

    fs::create_directories(results_dir);
    torch::save(model, results_dir + "/yolov1_bench_epoch.pt");

    state.SetItemsProcessed(total_processed);
    state.counters["Img/Sec"] = benchmark::Counter(
        static_cast<double>(total_processed),
        benchmark::Counter::kIsRate);
}

BENCHMARK(BM_YOLOv1_SingleEpochTraining)->Arg(8)->Arg(16)->UseRealTime();
BENCHMARK_MAIN();