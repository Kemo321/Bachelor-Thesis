#include <benchmark/benchmark.h>
#include <iostream>
#include <limits>
#include <torch/torch.h>

// Uwzględniamy Twoją strukturę folderów
#include "DeepLearnLib/dataset.hpp"
#include "DeepLearnLib/model.hpp"

static void BM_YOLOv1_SingleEpochTraining(benchmark::State& state)
{
    const int batch_size = state.range(0);
    const std::string data_root = "../data/VOCdevkit";

    // 1. Inicjalizacja sprzętu
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    
    // 2. Przygotowanie danych (VOC2012)
    DataPaths train_paths, val_paths, test_paths;
    split_dataset(data_root + "/VOC2012", train_paths, val_paths, test_paths);

    if (train_paths.images.empty()) {
        state.SkipWithError("Brak danych w JPEGImages/Annotations!");
        return;
    }

    auto train_loader = torch::data::make_data_loader(
        VOCYoloDataset(train_paths).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    // 3. Model i Optymalizator
    YOLOv1 model;
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));

    // Pętla benchmarka (wykona się dokładnie raz zgodnie z ustawieniem na dole)
    for (auto _ : state)
    {
        float epoch_loss = 0.0F;
        int processed_in_epoch = 0;
        model->train();

        for (auto& batch : *train_loader)
        {
            auto data = batch.data.to(device, true);
            auto target = batch.target.to(device, true);

            optimizer.zero_grad();
            auto pred = model->forward(data);
            auto loss = compute_yolo_loss(pred, target);

            loss.backward();
            optimizer.step();

            // Synchronizacja CUDA jest niezbędna do rzetelnego pomiaru czasu GPU
            if (device.is_cuda()) {
                torch::cuda::synchronize();
            }

            epoch_loss += loss.item<float>();
            processed_in_epoch += data.size(0);
        }

        // Dodajemy metrykę "Images Per Second" do raportu
        state.SetItemsProcessed(processed_in_epoch);
        benchmark::DoNotOptimize(epoch_loss);
    }

    // Zapis modelu po benchmarku (opcjonalne, do dokumentacji)
    torch::save(model, "yolov1_bench_epoch.pt");

    // Dodatkowe liczniki widoczne w tabeli wynikowej
    state.counters["Img/Sec"] = benchmark::Counter(
        static_cast<double>(train_paths.images.size()), 
        benchmark::Counter::kIsRate
    );
    state.counters["Batch"] = static_cast<double>(batch_size);
}

// Konfiguracja benchmarka pod RTX 5080
BENCHMARK(BM_YOLOv1_SingleEpochTraining)
    ->Arg(8)            // Batch size 8
    ->Arg(16)           // Batch size 16 (sprawdź ile VRAM zajmie 5080)
    ->Iterations(1)     // Dokładnie jedna epoka
    ->Unit(benchmark::kSecond)
    ->UseRealTime();

BENCHMARK_MAIN();