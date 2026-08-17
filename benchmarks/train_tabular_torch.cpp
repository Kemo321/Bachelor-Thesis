#include "experiment_config.hpp"
#include "run_metrics.hpp"
#include "tabular_common.hpp"

#include "DeepLearnLib/CSVLoader.hpp"
#include "DeepLearnLib/Logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <torch/torch.h>
#include <vector>

namespace fs = std::filesystem;

struct TabularMLPImpl : torch::nn::Module
{
    torch::nn::Linear fc1 { nullptr };
    torch::nn::Linear fc2 { nullptr };

    TabularMLPImpl(int in_features, int hidden, int out_features)
    {
        fc1 = register_module("fc1", torch::nn::Linear(in_features, hidden));
        fc2 = register_module("fc2", torch::nn::Linear(hidden, out_features));
    }

    auto forward(torch::Tensor input) -> torch::Tensor
    {
        return fc2->forward(torch::leaky_relu(fc1->forward(input), 0.1));
    }
};
TORCH_MODULE(TabularMLP);

int main()
{
    const nlohmann::json config = load_pipeline_config("tabular_demo");
    const int epochs = config.value("epochs", 20);
    const int batch_size = config.value("batch_size", 32);
    const float learning_rate = config.value("learning_rate", 0.05F);
    const int num_features = config.value("num_features", 4);
    const int hidden_size = config.value("hidden_size", 16);
    const int num_classes = config.value("num_classes", 3);
    const int num_samples = config.value("num_samples", 64);
    const bool skip_header = config.value("skip_header", true);
    const fs::path csv_path = resolve_from_source(config.value("csv_path", "data/tabular/demo.csv"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/tabular"));

    if (!fs::exists(csv_path))
    {
        LOG_INFO("[TABULAR TORCH] Writing dummy CSV at {}", csv_path.string());
        write_dummy_csv(csv_path, num_samples, num_features, num_classes, 42U);
    }

    CSVLoader loader(csv_path.string(), 1, skip_header);
    const int feature_count = loader.features().get_shape()[1];
    const int available = static_cast<int>(loader.size());
    const int batch = std::min(batch_size, available);
    LOG_INFO("[TABULAR TORCH] csv={} epochs={} batch_size={} lr={}", csv_path.string(), epochs, batch, learning_rate);

    std::vector<float> feature_host = loader.features().to_host();
    std::vector<float> label_host = loader.targets().to_host();
    feature_host.resize(static_cast<std::size_t>(batch) * static_cast<std::size_t>(feature_count));
    label_host.resize(static_cast<std::size_t>(batch));
    std::vector<int64_t> labels(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row)
    {
        labels[static_cast<std::size_t>(row)] =
            std::clamp(static_cast<int>(std::lround(label_host[static_cast<std::size_t>(row)])), 0, num_classes - 1);
    }

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    auto features = torch::from_blob(feature_host.data(), { batch, feature_count }, torch::kFloat32).clone().to(device);
    auto targets = torch::from_blob(labels.data(), { batch }, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);

    TabularMLP model(feature_count, hidden_size, num_classes);
    model->to(device);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));
    torch::nn::CrossEntropyLoss criterion;

    auto csv_file = open_metrics_csv(results_dir, "metrics_torch.csv", "Epoch;Loss;Time(s);VRAM_MiB;Acc");
    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        model->train();
        optimizer.zero_grad();
        auto logits = model->forward(features);
        auto loss = criterion(logits, targets);
        loss.backward();
        optimizer.step();
        const auto predicted = logits.argmax(1);
        const float accuracy = predicted.eq(targets).to(torch::kFloat).mean().item<float>();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        const auto vram = current_vram_mib();
        log_train_epoch("Tabular Torch", epoch, epochs, loss.item<float>(), elapsed, vram);
        LOG_INFO("Tabular Torch | Acc: {:.4f}", accuracy);
        csv_file << epoch << ";" << loss.item<float>() << ";" << elapsed << ";" << vram << ";" << accuracy << "\n";
        csv_file.flush();
    }
    return 0;
}
