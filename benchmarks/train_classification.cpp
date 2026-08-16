#include "experiment_config.hpp"

#include "DeepLearnLib/ClassificationLoader.hpp"
#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Profiler.hpp"
#include "DeepLearnLib/SimpleCNN.hpp"

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

auto argmax_row(const std::vector<float>& values, int row, int cols) -> int
{
    const std::size_t offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    int best = 0;
    float best_value = values[offset];
    for (int col = 1; col < cols; ++col)
    {
        const float value = values[offset + static_cast<std::size_t>(col)];
        if (value > best_value)
        {
            best_value = value;
            best = col;
        }
    }
    return best;
}

auto batch_accuracy(const dl::Tensor& logits, const dl::Tensor& one_hot) -> float
{
    const std::vector<float> logit_host = logits.to_host();
    const std::vector<float> target_host = one_hot.to_host();
    const int batch = logits.get_shape()[0];
    const int classes = logits.get_shape()[1];
    int correct = 0;
    for (int row = 0; row < batch; ++row)
    {
        if (argmax_row(logit_host, row, classes) == argmax_row(target_host, row, classes))
        {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(std::max(1, batch));
}

int main()
{
    const nlohmann::json config = load_pipeline_config("cifar10_classification");
    const int batch_size = config.value("batch_size", 64);
    const int total_epochs = config.value("epochs", 20);
    const float learning_rate = config.value("learning_rate", 1.0e-3F);
    const int image_size = config.value("image_size", 32);
    const std::string train_split = config.value("train_split", "train");
    const std::string test_split = config.value("test_split", "test");
    const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/cifar10"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/cifar10"));

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    std::cout << "[CIFAR-10 CLASSIFICATION] Starting on device: " << (gpu_count > 0 ? "GPU" : "CPU") << "\n";
    std::cout << "[CONFIG] batch_size=" << batch_size << " epochs=" << total_epochs
              << " learning_rate=" << learning_rate << " dataset_root=" << data_root << "\n";

    ClassificationLoader train_loader(data_root.string(), train_split, batch_size, image_size, true);
    ClassificationLoader test_loader(data_root.string(), test_split, batch_size, image_size, false);
    const int num_classes = train_loader.num_classes();
    std::cout << "[CONFIG] classes=" << num_classes << " train=" << train_loader.size()
              << " test=" << test_loader.size() << "\n";

    SimpleCNN model(num_classes, image_size);
    Network trainer(model.get_all_layers(), learning_rate);
    for (auto& layer : model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->learning_rate = learning_rate;
        layer->train();
    }

    fs::create_directories(results_dir);
    std::ofstream csv_file((results_dir / "metrics_custom.csv").string());
    csv_file << "Epoch;TrainLoss;TestLoss;TrainAcc;TestAcc;Time(s)\n";

    Profiler profiler;
    for (int epoch = 1; epoch <= total_epochs; ++epoch)
    {
        auto epoch_start = std::chrono::steady_clock::now();
        profiler.start();

        for (auto& layer : model.get_all_layers())
        {
            layer->train();
            layer->learning_rate = learning_rate;
        }

        float train_loss = 0.0F;
        float train_acc = 0.0F;
        int train_batches = 0;
        train_loader.reset();
        while (train_loader.has_next())
        {
            Batch batch = train_loader.get_batch();
            dl::Tensor logits = model.forward_logits(batch.images);
            train_loss += CrossEntropyLoss::loss(batch.targets, logits).to_host().front();
            train_acc += batch_accuracy(logits, batch.targets);

            dl::Tensor grad = CrossEntropyLoss::loss_derivative(batch.targets, logits);
            auto layers = model.get_all_layers();
            for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
            {
                grad = (*iterator)->backward(grad);
            }
            for (auto& layer : layers)
            {
                layer->step();
            }
            ++train_batches;
        }

        for (auto& layer : model.get_all_layers())
        {
            layer->eval();
        }

        float test_loss = 0.0F;
        float test_acc = 0.0F;
        int test_batches = 0;
        test_loader.reset();
        while (test_loader.has_next())
        {
            Batch batch = test_loader.get_batch();
            dl::Tensor logits = model.forward_logits(batch.images);
            test_loss += CrossEntropyLoss::loss(batch.targets, logits).to_host().front();
            test_acc += batch_accuracy(logits, batch.targets);
            ++test_batches;
        }

        const float gpu_ms = profiler.stop();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - epoch_start)
                                 .count();
        const float avg_train = train_loss / static_cast<float>(std::max(1, train_batches));
        const float avg_test = test_loss / static_cast<float>(std::max(1, test_batches));
        const float avg_train_acc = train_acc / static_cast<float>(std::max(1, train_batches));
        const float avg_test_acc = test_acc / static_cast<float>(std::max(1, test_batches));

        std::cout << "CIFAR-10 | Epoch [" << std::setw(3) << epoch << "/" << total_epochs << "] | Train Loss: "
                  << std::fixed << std::setprecision(4) << avg_train << " | Test Loss: " << avg_test
                  << " | Train Acc: " << avg_train_acc << " | Test Acc: " << avg_test_acc << " | Time: " << elapsed
                  << "s | GPU: " << gpu_ms << " ms\n";
        csv_file << epoch << ";" << avg_train << ";" << avg_test << ";" << avg_train_acc << ";" << avg_test_acc << ";"
                 << elapsed << "\n";
        csv_file.flush();
    }

    const std::string save_path = (results_dir / "simplecnn_cifar10_final.bin").string();
    trainer.save(save_path);
    std::cout << "[INFO] Final model saved: " << save_path << "\n";
    return 0;
}
