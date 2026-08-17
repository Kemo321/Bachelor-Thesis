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
#include <string>
#include <vector>

namespace fs = std::filesystem;

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
        LOG_INFO("[TABULAR CUSTOM] Writing dummy CSV at {}", csv_path.string());
        write_dummy_csv(csv_path, num_samples, num_features, num_classes, 42U);
    }

    CSVLoader loader(csv_path.string(), 1, skip_header);
    const int feature_count = loader.features().get_shape()[1];
    const int available = static_cast<int>(loader.size());
    const int batch = std::min(batch_size, available);
    LOG_INFO("[TABULAR CUSTOM] csv={} epochs={} batch_size={} lr={}", csv_path.string(), epochs, batch, learning_rate);

    std::vector<float> feature_host = loader.features().to_host();
    std::vector<float> label_host = loader.targets().to_host();
    feature_host.resize(static_cast<std::size_t>(batch) * static_cast<std::size_t>(feature_count));
    label_host.resize(static_cast<std::size_t>(batch));
    dl::Tensor features = dl::Tensor::from_host({ batch, feature_count }, feature_host, dl::Device::GPU);
    const std::vector<float> target_host = one_hot_labels(label_host, num_classes);
    dl::Tensor targets = dl::Tensor::from_host({ batch, num_classes }, target_host, dl::Device::GPU);

    auto dense1 = std::make_shared<FullyConnected>(feature_count, hidden_size, 0.0F);
    auto relu = std::make_shared<LeakyReLU>(0.1F);
    auto dense2 = std::make_shared<FullyConnected>(hidden_size, num_classes, 0.0F);
    auto softmax = std::make_shared<Softmax>();
    std::vector<std::shared_ptr<Layer>> layers = { dense1, relu, dense2 };
    for (auto& layer : layers)
    {
        layer->learning_rate = learning_rate;
        layer->to(dl::Device::GPU);
        layer->train();
    }
    softmax->to(dl::Device::GPU);
    softmax->eval();

    auto csv_file = open_metrics_csv(results_dir, "metrics_custom.csv", "Epoch;Loss;Time(s);VRAM_MiB;Acc");
    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        const auto epoch_start = std::chrono::steady_clock::now();
        dl::Tensor hidden = relu->forward(dense1->forward(features));
        dl::Tensor logits = dense2->forward(hidden);
        const float loss = CrossEntropyLoss::loss(targets, logits).to_host().front();
        dl::Tensor probabilities = softmax->forward(logits);
        const std::vector<float> prob_host = probabilities.to_host();
        int correct = 0;
        for (int row = 0; row < batch; ++row)
        {
            if (argmax_row(prob_host, row, num_classes)
                == static_cast<int>(std::lround(label_host[static_cast<std::size_t>(row)])))
            {
                ++correct;
            }
        }
        const float accuracy = static_cast<float>(correct) / static_cast<float>(batch);

        dl::Tensor grad = CrossEntropyLoss::loss_derivative(targets, logits);
        for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
        {
            grad = (*iterator)->backward(grad);
        }
        for (auto& layer : layers)
        {
            layer->step();
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
        const auto vram = current_vram_mib();
        log_train_epoch("Tabular Custom", epoch, epochs, loss, elapsed, vram);
        LOG_INFO("Tabular Custom | Acc: {:.4f}", accuracy);
        csv_file << epoch << ";" << loss << ";" << elapsed << ";" << vram << ";" << accuracy << "\n";
        csv_file.flush();
    }
    return 0;
}
