#include "experiment_config.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/ClassificationLoader.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct SimpleCNNImpl : torch::nn::Module
{
    torch::nn::Conv2d conv1 { nullptr };
    torch::nn::Conv2d conv2 { nullptr };
    torch::nn::Linear fc { nullptr };

    SimpleCNNImpl(int num_classes, int image_size)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)));
        const int spatial = image_size / 4;
        fc = register_module("fc", torch::nn::Linear(32 * spatial * spatial, num_classes));
    }

    auto forward(torch::Tensor input) -> torch::Tensor
    {
        auto hidden = torch::leaky_relu(conv1->forward(input), 0.1);
        hidden = torch::max_pool2d(hidden, 2);
        hidden = torch::leaky_relu(conv2->forward(hidden), 0.1);
        hidden = torch::max_pool2d(hidden, 2);
        return fc->forward(hidden.flatten(1));
    }
};
TORCH_MODULE(SimpleCNN);

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

auto batch_to_torch(const Batch& batch, const torch::Device& device) -> std::pair<torch::Tensor, torch::Tensor>
{
    const auto& image_shape = batch.images.get_shape();
    const auto& target_shape = batch.targets.get_shape();
    const int n = image_shape[0];
    const int classes = target_shape[1];
    std::vector<float> image_host = batch.images.to_host();
    std::vector<float> target_host = batch.targets.to_host();
    auto images = torch::from_blob(image_host.data(), { n, image_shape[1], image_shape[2], image_shape[3] }, torch::kFloat32)
                      .clone()
                      .to(device);
    std::vector<int64_t> labels(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row)
    {
        labels[static_cast<std::size_t>(row)] = argmax_row(target_host, row, classes);
    }
    auto targets = torch::from_blob(labels.data(), { n }, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
    return { images, targets };
}

int main()
{
    try
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

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        LOG_INFO("[CIFAR-10 TORCH] Starting on device: {}", device.is_cuda() ? "GPU" : "CPU");
        LOG_INFO("[CONFIG] batch_size={} epochs={} learning_rate={} dataset_root={}", batch_size, total_epochs,
            learning_rate, data_root.string());

        ClassificationLoader train_loader(data_root.string(), train_split, batch_size, image_size, true);
        const std::vector<std::string> class_names = train_loader.class_names();
        ClassificationLoader test_loader(data_root.string(), test_split, batch_size, image_size, false, class_names);
        const int num_classes = train_loader.num_classes();
        if (test_loader.num_classes() != num_classes)
        {
            throw std::runtime_error("CIFAR train/test class counts differ");
        }
        LOG_INFO("[CONFIG] classes={} train={} test={}", num_classes, train_loader.size(), test_loader.size());

        SimpleCNN model(num_classes, image_size);
        model->to(device);
        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));
        torch::nn::CrossEntropyLoss criterion;

        auto csv_file = open_metrics_csv(
            results_dir, "metrics_torch.csv", "Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;TrainAcc;TestAcc");

        for (int epoch = 1; epoch <= total_epochs; ++epoch)
        {
            const auto epoch_start = std::chrono::steady_clock::now();
            model->train();
            float train_loss = 0.0F;
            float train_acc = 0.0F;
            int train_batches = 0;
            train_loader.reset();
            while (train_loader.has_next())
            {
                Batch batch = train_loader.get_batch();
                auto tensors = batch_to_torch(batch, device);
                optimizer.zero_grad();
                auto logits = model->forward(tensors.first);
                auto loss = criterion(logits, tensors.second);
                loss.backward();
                optimizer.step();
                train_loss += loss.item<float>();
                const auto predicted = logits.argmax(1);
                train_acc += predicted.eq(tensors.second).to(torch::kFloat).mean().item<float>();
                ++train_batches;
            }

            model->eval();
            float test_loss = 0.0F;
            float test_acc = 0.0F;
            int test_batches = 0;
            {
                torch::NoGradGuard no_grad;
                test_loader.reset();
                while (test_loader.has_next())
                {
                    Batch batch = test_loader.get_batch();
                    auto tensors = batch_to_torch(batch, device);
                    auto logits = model->forward(tensors.first);
                    test_loss += criterion(logits, tensors.second).item<float>();
                    const auto predicted = logits.argmax(1);
                    test_acc += predicted.eq(tensors.second).to(torch::kFloat).mean().item<float>();
                    ++test_batches;
                }
            }

            const float avg_train = train_loss / static_cast<float>(std::max(1, train_batches));
            const float avg_test = test_loss / static_cast<float>(std::max(1, test_batches));
            const float avg_train_acc = train_acc / static_cast<float>(std::max(1, train_batches));
            const float avg_test_acc = test_acc / static_cast<float>(std::max(1, test_batches));
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - epoch_start).count();
            const auto vram = current_vram_mib();
            log_train_epoch("CIFAR-10 Torch", epoch, total_epochs, avg_train, avg_test, elapsed, vram);
            LOG_INFO("CIFAR-10 Torch | Train Acc: {:.4f} | Test Acc: {:.4f}", avg_train_acc, avg_test_acc);
            csv_file << epoch << ";" << avg_train << ";" << avg_test << ";" << elapsed << ";" << vram << ";"
                     << avg_train_acc << ";" << avg_test_acc << "\n";
            csv_file.flush();
        }

        const std::string save_path = (results_dir / "simplecnn_cifar10_torch_final.pt").string();
        torch::save(model, save_path);
        LOG_INFO("Final model saved: {}", save_path);
        return 0;
    }
    catch (const std::exception& exception)
    {
        LOG_ERROR("CIFAR-10 Torch classification failed: {}", exception.what());
        return 1;
    }
}
