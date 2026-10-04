#include "classification_eval.hpp"
#include "experiment_config.hpp"
#include "run_metrics.hpp"
#include "tabular_common.hpp"

#include "DeepLearnLib/CSVLoader.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/Softmax.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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
    // Read the selected pipeline JSON so the epochs, hidden-layer size, and CSV path come from the experiment.
    const nlohmann::json config = load_pipeline_config(pipeline);
    const int epochs = config.value("epochs", 20);
    const int batch_size = config.value("batch_size", 32);
    const float learning_rate = config.value("learning_rate", 0.05F);
    const float momentum = config.value("momentum", 0.9F);
    const float weight_decay = config.value("weight_decay", 0.0005F);
    const float gradient_clip = pipeline_gradient_clip(config);
    const int hidden_size = config.value("hidden_size", 16);
    const int num_classes = config.value("num_classes", 3);
    const int num_samples = config.value("num_samples", 64);
    const int num_features_cfg = config.value("num_features", 4);
    const bool skip_header = config.value("skip_header", true);
    const fs::path csv_path = resolve_from_source(config.value("csv_path", "data/tabular/demo.csv"));
    const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/tabular"));
    // Keep the console log next to the CSV. A new run replaces this file.
    open_results_log(results_dir, "log_custom.txt");
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

    // Load the whole CSV table for training so the experiment measures fit on every row.
    if (!fs::exists(csv_path))
    {
        LOG_INFO("[TABULAR CUSTOM] Writing dummy CSV at {}", csv_path.string());
        write_dummy_csv(csv_path, num_samples, num_features_cfg, num_classes, 42U);
    }

    CSVLoader loader(csv_path.string(), 1, skip_header);
    const int feature_count = loader.features().get_shape()[1];
    const int available = static_cast<int>(loader.size());
    const int batch = std::max(1, std::min(batch_size, available));

    std::vector<float> feature_host = loader.features().to_host();
    std::vector<float> label_host = loader.targets().to_host();

    // Two dense layers with LeakyReLU on the GPU, because the input is a feature vector from the table.
    auto dense1 = std::make_shared<FullyConnected>(feature_count, hidden_size, 0.0F);
    auto relu = std::make_shared<LeakyReLU>(0.1F);
    auto dense2 = std::make_shared<FullyConnected>(hidden_size, num_classes, 0.0F);
    auto softmax = std::make_shared<Softmax>();
    std::vector<std::shared_ptr<Layer>> layers = { dense1, relu, dense2 };
    for (auto& layer : layers)
    {
        layer->to(dl::Device::GPU);
        layer->train();
    }
    apply_sgd_hyperparameters(layers, learning_rate, momentum, weight_decay);
    for (auto& layer : layers)
    {
        layer->gradient_clip = gradient_clip;
    }
    softmax->to(dl::Device::GPU);
    softmax->eval();

    log_pipeline_banner({ "Tabular Custom", "custom", batch, epochs, learning_rate, momentum, weight_decay,
        gradient_clip, num_classes, 0, static_cast<int>(layers.size()),
        csv_path.string(), static_cast<std::size_t>(available), 0 });
    LOG_INFO("Tabular Custom | pipeline={}", pipeline);

    write_class_names(results_dir / "class_names.txt", class_names);
    // Metrics CSV so each epoch appends loss, time, and accuracy for comparing runs.
    auto csv_file = open_metrics_csv(results_dir, "metrics_custom.csv", kTabularCsvHeader);
    std::mt19937 rng(42U);
    std::vector<int> order(static_cast<std::size_t>(available));
    std::iota(order.begin(), order.end(), 0);
    std::vector<int> confusion;

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        const float current_lr = scheduled_learning_rate(config, epoch);
        apply_sgd_hyperparameters(layers, current_lr, momentum, weight_decay);
        // The epoch shuffles the rows and updates the weights on every batch so the CSV order does not stick in the gradient.
        std::shuffle(order.begin(), order.end(), rng);
        float epoch_loss = 0.0F;
        int epoch_correct = 0;
        int epoch_seen = 0;
        int batches = 0;
        confusion.assign(static_cast<std::size_t>(num_classes) * static_cast<std::size_t>(num_classes), 0);

        for (int start = 0; start < available; start += batch)
        {
            const int n = std::min(batch, available - start);
            std::vector<float> batch_features(static_cast<std::size_t>(n) * static_cast<std::size_t>(feature_count));
            std::vector<float> batch_labels(static_cast<std::size_t>(n));
            for (int row = 0; row < n; ++row)
            {
                const int sample = order[static_cast<std::size_t>(start + row)];
                std::copy_n(feature_host.data() + (static_cast<std::size_t>(sample) * static_cast<std::size_t>(feature_count)),
                    static_cast<std::size_t>(feature_count),
                    batch_features.data() + (static_cast<std::size_t>(row) * static_cast<std::size_t>(feature_count)));
                batch_labels[static_cast<std::size_t>(row)] = label_host[static_cast<std::size_t>(sample)];
            }
            dl::Tensor features = dl::Tensor::from_host({ n, feature_count }, batch_features, dl::Device::GPU);
            dl::Tensor targets
                = dl::Tensor::from_host({ n, num_classes }, one_hot_labels(batch_labels, num_classes), dl::Device::GPU);

            dl::Tensor hidden = relu->forward(dense1->forward(features));
            dl::Tensor logits = dense2->forward(hidden);
            epoch_loss += CrossEntropyLoss::loss(targets, logits).to_host().front();
            // Accuracy and the confusion matrix on the current batch, so the epoch reports fit to the same table.
            // Accuracy is on the same rows used for the step, because there is no held-out split.
            dl::Tensor probabilities = softmax->forward(logits);
            const std::vector<float> prob_host = probabilities.to_host();
            std::vector<int> truths(static_cast<std::size_t>(n));
            std::vector<int> preds(static_cast<std::size_t>(n));
            for (int row = 0; row < n; ++row)
            {
                truths[static_cast<std::size_t>(row)] = std::clamp(
                    static_cast<int>(std::lround(batch_labels[static_cast<std::size_t>(row)])), 0, num_classes - 1);
                preds[static_cast<std::size_t>(row)] = argmax_row(prob_host, row, num_classes);
                if (preds[static_cast<std::size_t>(row)] == truths[static_cast<std::size_t>(row)])
                {
                    ++epoch_correct;
                }
            }
            accumulate_confusion_ids(confusion, truths, preds, num_classes);
            epoch_seen += n;
            ++batches;

            dl::Tensor grad = CrossEntropyLoss::loss_derivative(targets, logits);
            for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
            {
                grad = (*iterator)->backward(grad);
            }
            for (auto& layer : layers)
            {
                layer->step();
            }
        }

        const float avg_loss = epoch_loss / static_cast<float>(std::max(1, batches));
        const float accuracy = static_cast<float>(epoch_correct) / static_cast<float>(std::max(1, epoch_seen));
        const auto elapsed
            = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        log_train_epoch({ "Tabular Custom", epoch, epochs, current_lr, avg_loss, std::nullopt, accuracy, std::nullopt,
            std::nullopt, batches, elapsed, current_vram_mib() });
        write_tabular_row(csv_file, epoch, avg_loss, elapsed, current_vram_mib(), accuracy);
    }

    write_confusion_csv(results_dir / "confusion_custom.csv", confusion, num_classes, class_names);
    return 0;
}
