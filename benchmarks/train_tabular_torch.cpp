#include "classification_eval.hpp"
#include "experiment_config.hpp"
#include "run_metrics.hpp"
#include "tabular_common.hpp"

#include "DeepLearnLib/CSVLoader.hpp"
#include "DeepLearnLib/Logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
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

auto pipeline_name_from_args(int argc, char** argv) -> std::string
{
    if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0')
    {
        return argv[1];
    }
    return "tabular_demo";
}

int main(int argc, char** argv)
{
    const std::string pipeline = pipeline_name_from_args(argc, argv);
    const nlohmann::json config = load_pipeline_config(pipeline);
    const int epochs = config.value("epochs", 20);
    const int batch_size = config.value("batch_size", 32);
    const float learning_rate = config.value("learning_rate", 0.05F);
    const double momentum = config.value("momentum", 0.9);
    const double weight_decay = config.value("weight_decay", 0.0005);
    const int hidden_size = config.value("hidden_size", 16);
    const int num_classes = config.value("num_classes", 3);
    const int num_samples = config.value("num_samples", 64);
    const int num_features_cfg = config.value("num_features", 4);
    const bool skip_header = config.value("skip_header", true);
    const fs::path csv_path = resolve_from_source(config.value("csv_path", "data/tabular/demo.csv"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/tabular"));
    std::vector<std::string> class_names;
    if (config.contains("class_names") && config.at("class_names").is_array())
    {
        class_names = config.at("class_names").get<std::vector<std::string>>();
    }
    if (class_names.empty())
    {
        for (int class_id = 0; class_id < num_classes; ++class_id)
        {
            class_names.push_back(std::to_string(class_id));
        }
    }

    if (!fs::exists(csv_path))
    {
        LOG_INFO("[TABULAR TORCH] Writing dummy CSV at {}", csv_path.string());
        write_dummy_csv(csv_path, num_samples, num_features_cfg, num_classes, 42U);
    }

    CSVLoader loader(csv_path.string(), 1, skip_header);
    const int feature_count = loader.features().get_shape()[1];
    const int available = static_cast<int>(loader.size());
    const int batch = std::max(1, std::min(batch_size, available));
    LOG_INFO("[TABULAR TORCH] pipeline={} csv={} epochs={} batch_size={} lr={} momentum={} weight_decay={} n={}",
        pipeline, csv_path.string(), epochs, batch, learning_rate, momentum, weight_decay, available);

    std::vector<float> feature_host = loader.features().to_host();
    std::vector<float> label_host = loader.targets().to_host();
    auto get_lr = [&config](int ep) -> float
    { return scheduled_learning_rate(config, ep); };

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    TabularMLP model(feature_count, hidden_size, num_classes);
    model->to(device);
    torch::optim::SGD optimizer(model->parameters(),
        torch::optim::SGDOptions(get_lr(1)).momentum(momentum).weight_decay(weight_decay));
    torch::nn::CrossEntropyLoss criterion;

    write_class_names(results_dir / "class_names.txt", class_names);
    auto csv_file = open_metrics_csv(results_dir, "metrics_torch.csv", "Epoch;Loss;Time(s);VRAM_MiB;Acc");
    std::mt19937 rng(42U);
    std::vector<int> order(static_cast<std::size_t>(available));
    std::iota(order.begin(), order.end(), 0);
    std::vector<int> confusion;

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_lr = get_lr(epoch);
        for (auto& group : optimizer.param_groups())
        {
            static_cast<torch::optim::SGDOptions&>(group.options()).lr(current_lr);
        }
        std::shuffle(order.begin(), order.end(), rng);
        model->train();
        float epoch_loss = 0.0F;
        int epoch_correct = 0;
        int epoch_seen = 0;
        int batches = 0;
        confusion.assign(static_cast<std::size_t>(num_classes) * static_cast<std::size_t>(num_classes), 0);

        for (int start = 0; start < available; start += batch)
        {
            const int n = std::min(batch, available - start);
            std::vector<float> batch_features(static_cast<std::size_t>(n) * static_cast<std::size_t>(feature_count));
            std::vector<int64_t> labels(static_cast<std::size_t>(n));
            for (int row = 0; row < n; ++row)
            {
                const int sample = order[static_cast<std::size_t>(start + row)];
                std::copy_n(feature_host.data() + (static_cast<std::size_t>(sample) * static_cast<std::size_t>(feature_count)),
                    static_cast<std::size_t>(feature_count),
                    batch_features.data() + (static_cast<std::size_t>(row) * static_cast<std::size_t>(feature_count)));
                labels[static_cast<std::size_t>(row)] = std::clamp(
                    static_cast<int>(std::lround(label_host[static_cast<std::size_t>(sample)])), 0, num_classes - 1);
            }
            auto features
                = torch::from_blob(batch_features.data(), { n, feature_count }, torch::kFloat32).clone().to(device);
            auto targets
                = torch::from_blob(labels.data(), { n }, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);

            optimizer.zero_grad();
            auto logits = model->forward(features);
            auto loss = criterion(logits, targets);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<float>();
            const auto predicted = logits.argmax(1);
            epoch_correct += static_cast<int>(predicted.eq(targets).sum().item<int64_t>());
            epoch_seen += n;
            ++batches;

            const auto pred_cpu = predicted.to(torch::kCPU);
            std::vector<int> truths(static_cast<std::size_t>(n));
            std::vector<int> preds(static_cast<std::size_t>(n));
            for (int row = 0; row < n; ++row)
            {
                truths[static_cast<std::size_t>(row)] = static_cast<int>(labels[static_cast<std::size_t>(row)]);
                preds[static_cast<std::size_t>(row)] = static_cast<int>(pred_cpu[row].item<int64_t>());
            }
            accumulate_confusion_ids(confusion, truths, preds, num_classes);
        }

        const float avg_loss = epoch_loss / static_cast<float>(std::max(1, batches));
        const float accuracy = static_cast<float>(epoch_correct) / static_cast<float>(std::max(1, epoch_seen));
        const auto elapsed
            = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        const auto vram = current_vram_mib();
        log_train_epoch("Tabular Torch", epoch, epochs, avg_loss, elapsed, vram);
        LOG_INFO("Tabular Torch | pipeline={} Acc: {:.4f}", pipeline, accuracy);
        csv_file << epoch << ";" << avg_loss << ";" << elapsed << ";" << vram << ";" << accuracy << "\n";
        csv_file.flush();
    }

    write_confusion_csv(results_dir / "confusion_torch.csv", confusion, num_classes, class_names);
    return 0;
}
