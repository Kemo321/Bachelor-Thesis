#include "experiment_config.hpp"

#include "DeepLearnLib/CSVLoader.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/Layer.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/Profiler.hpp"
#include "DeepLearnLib/Softmax.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

auto write_dummy_csv(const fs::path& csv_path, int num_samples, int num_features, int num_classes, unsigned seed)
    -> void
{
    fs::create_directories(csv_path.parent_path());
    std::ofstream stream(csv_path);
    if (!stream)
    {
        throw std::runtime_error("Could not write dummy CSV: " + csv_path.string());
    }

    for (int feature = 0; feature < num_features; ++feature)
    {
        stream << "f" << feature << ",";
    }
    stream << "label\n";

    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0F, 0.15F);
    for (int row = 0; row < num_samples; ++row)
    {
        const int label = row % num_classes;
        for (int feature = 0; feature < num_features; ++feature)
        {
            const float value = (feature == label ? 1.0F : 0.0F) + noise(rng);
            stream << value << ",";
        }
        stream << label << "\n";
    }
}

auto one_hot(const std::vector<float>& class_ids, int num_classes) -> std::vector<float>
{
    std::vector<float> encoded(class_ids.size() * static_cast<std::size_t>(num_classes), 0.0F);
    for (std::size_t row = 0; row < class_ids.size(); ++row)
    {
        int label = static_cast<int>(std::lround(class_ids[row]));
        label = std::clamp(label, 0, num_classes - 1);
        encoded[(row * static_cast<std::size_t>(num_classes)) + static_cast<std::size_t>(label)] = 1.0F;
    }
    return encoded;
}

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

    if (!fs::exists(csv_path))
    {
        LOG_INFO("[tabular_demo] Writing dummy CSV at {}", csv_path.string());
        write_dummy_csv(csv_path, num_samples, num_features, num_classes, 42U);
    }

    CSVLoader loader(csv_path.string(), 1, skip_header);
    const int feature_count = loader.features().get_shape()[1];
    const int available = static_cast<int>(loader.size());
    const int batch = std::min(batch_size, available);
    LOG_INFO("[tabular_demo] csv={} epochs={} batch_size={} lr={}", csv_path.string(), epochs, batch, learning_rate);
    std::vector<float> feature_host = loader.features().to_host();
    std::vector<float> label_host = loader.targets().to_host();
    feature_host.resize(static_cast<std::size_t>(batch) * static_cast<std::size_t>(feature_count));
    label_host.resize(static_cast<std::size_t>(batch));
    dl::Tensor features = dl::Tensor::from_host({ batch, feature_count }, feature_host, dl::Device::GPU);
    const std::vector<float> target_host = one_hot(label_host, num_classes);
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

    Profiler profiler;
    profiler.start();
    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
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

        LOG_INFO("Tabular | Epoch [{}/{}] | CE: {} | Acc: {}", epoch, epochs, loss, accuracy);
    }
    const float gpu_ms = profiler.stop();
    LOG_INFO("[tabular_demo] GPU time: {} ms | VRAM: {} MiB", gpu_ms, Profiler::get_vram_usage_mb());
    return 0;
}
