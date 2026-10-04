#include "classification_eval.hpp"
#include "classification_vis.hpp"
#include "experiment_config.hpp"
#include "run_metrics.hpp"
#include "torch_optim.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/PackedImageLoader.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <optional>
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

    SimpleCNNImpl(int num_classes, int image_size, int in_channels)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 16, 3).padding(1)));
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

auto batch_to_torch(const Batch& batch, const torch::Device& device) -> std::pair<torch::Tensor, torch::Tensor>
{
    const auto& image_shape = batch.images.get_shape();
    const auto& target_shape = batch.targets.get_shape();
    const int n = image_shape[0];
    const int classes = target_shape[1];
    std::vector<float> image_host = batch.images.to_host();
    std::vector<float> target_host = batch.targets.to_host();
    auto images = torch::from_blob(
        image_host.data(), { n, image_shape[1], image_shape[2], image_shape[3] }, torch::kFloat32)
                      .clone()
                      .to(device);
    std::vector<int64_t> labels(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row)
    {
        labels[static_cast<std::size_t>(row)] = classification_argmax_row(target_host, row, classes);
    }
    auto targets = torch::from_blob(labels.data(), { n }, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
    return { images, targets };
}

int main()
{
    try
    {
        // Read the mnist_classification pipeline JSON so the epoch count, learning rate, and paths come from the experiment.
        const nlohmann::json config = load_pipeline_config("mnist_classification");
        const int batch_size = config.value("batch_size", 128);
        const int total_epochs = config.value("epochs", 10);
        const float learning_rate = config.value("learning_rate", 1.0e-2F);
        const double momentum = config.value("momentum", 0.9);
        const double weight_decay = config.value("weight_decay", 0.0005);
        const float gradient_clip = pipeline_gradient_clip(config);
        const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/mnist"));
        const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/mnist"));
        // Keep the console log next to the CSV. A new run replaces this file.
        open_results_log(results_dir, "log_torch.txt");
        const fs::path train_bin = data_root / "train.bin";
        const fs::path test_bin = data_root / "test.bin";

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

        // Separate train and test files so accuracy after the epoch is computed on images held out of training.
        PackedImageLoader train_loader(train_bin.string(), batch_size, true);
        PackedImageLoader test_loader(test_bin.string(), batch_size, false);
        const int num_classes = train_loader.num_classes();
        const int image_size = train_loader.height();
        const int in_channels = train_loader.channels();

        auto get_lr = [&config](int ep) -> float
        { return scheduled_learning_rate(config, ep); };

        // Build SimpleCNN in LibTorch so the same classifier is trained on the reference stack.
        SimpleCNN model(num_classes, image_size, in_channels);
        model->to(device);
        torch::optim::SGD optimizer = make_sgd(*model, get_lr(1), momentum, weight_decay);
        torch::nn::CrossEntropyLoss criterion;

        log_pipeline_banner({ "MNIST Torch", "torch", batch_size, total_epochs, learning_rate,
            static_cast<float>(momentum), static_cast<float>(weight_decay), gradient_clip,
            num_classes, 0, 0, train_bin.string(),
            static_cast<std::size_t>(train_loader.size()), static_cast<std::size_t>(test_loader.size()) });

        write_class_names(results_dir / "class_names.txt", train_loader.class_names());
        // Metrics CSV so each epoch appends a row for comparing runs later.
        auto csv_file = open_metrics_csv(
            results_dir, "metrics_torch.csv", kClassificationCsvHeader);

        std::vector<int> confusion;
        std::vector<SamplePrediction> samples;
        for (int epoch = 1; epoch <= total_epochs; ++epoch)
        {
            const auto epoch_start = std::chrono::steady_clock::now();
            const float current_lr = get_lr(epoch);
            set_sgd_lr(optimizer, current_lr);
            // Training epoch in train mode: cross-entropy and an SGD step so the weights fit the training images.
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
                clip_torch_grad_value(*model, gradient_clip);
                optimizer.step();
                train_loss += loss.item<float>();
                train_acc += logits.argmax(1).eq(tensors.second).to(torch::kFloat).mean().item<float>();
                ++train_batches;
            }

            // Test pass in eval mode, to measure loss and accuracy on the test set.
            model->eval();
            float test_loss = 0.0F;
            float test_acc = 0.0F;
            int test_batches = 0;
            confusion.assign(static_cast<std::size_t>(num_classes) * static_cast<std::size_t>(num_classes), 0);
            samples.clear();
            int seen_eval = 0;
            {
                torch::NoGradGuard no_grad;
                test_loader.reset();
                while (test_loader.has_next())
                {
                    Batch batch = test_loader.get_batch();
                    auto tensors = batch_to_torch(batch, device);
                    auto logits = model->forward(tensors.first);
                    test_loss += criterion(logits, tensors.second).item<float>();
                    test_acc += logits.argmax(1).eq(tensors.second).to(torch::kFloat).mean().item<float>();
                    ++test_batches;
                    if (epoch == total_epochs)
                    {
                        const auto pred_cpu = logits.argmax(1).to(torch::kCPU);
                        const auto truth_cpu = tensors.second.to(torch::kCPU);
                        const auto prob_cpu = torch::softmax(logits, 1).to(torch::kCPU);
                        const int n = batch.images.get_shape()[0];
                        std::vector<int> truths(static_cast<std::size_t>(n));
                        std::vector<int> preds(static_cast<std::size_t>(n));
                        std::vector<float> confidences(static_cast<std::size_t>(n));
                        for (int row = 0; row < n; ++row)
                        {
                            truths[static_cast<std::size_t>(row)] = static_cast<int>(truth_cpu[row].item<int64_t>());
                            preds[static_cast<std::size_t>(row)] = static_cast<int>(pred_cpu[row].item<int64_t>());
                            confidences[static_cast<std::size_t>(row)]
                                = prob_cpu[row][preds[static_cast<std::size_t>(row)]].item<float>();
                        }
                        accumulate_confusion_ids(confusion, truths, preds, num_classes);
                        append_samples_from_ids(batch.images, truths, preds, confidences, seen_eval, 24, samples);
                    }
                    seen_eval += batch.images.get_shape()[0];
                }
            }

            const float avg_train = train_loss / static_cast<float>(std::max(1, train_batches));
            const float avg_test = test_loss / static_cast<float>(std::max(1, test_batches));
            const float avg_train_acc = train_acc / static_cast<float>(std::max(1, train_batches));
            const float avg_test_acc = test_acc / static_cast<float>(std::max(1, test_batches));
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - epoch_start)
                                     .count();
            log_train_epoch({ "MNIST Torch", epoch, total_epochs, current_lr, avg_train, avg_test, avg_train_acc,
                avg_test_acc, std::nullopt, train_batches, elapsed, current_vram_mib() });
            write_classification_row(csv_file, epoch, avg_train, avg_test, elapsed, current_vram_mib(), avg_train_acc,
                avg_test_acc);
        }

        write_confusion_csv(results_dir / "confusion_torch.csv", confusion, num_classes, train_loader.class_names());
        write_classification_samples(results_dir / "samples_torch", samples, train_loader.class_names());
        // Save the weights at the end of the run so the same model can be loaded without repeating training.
        const std::string save_path = (results_dir / "simplecnn_mnist_torch_final.pt").string();
        torch::save(model, save_path);
        log_saved("MNIST Torch", save_path);
        return 0;
    }
    catch (const std::exception& exception)
    {
        LOG_ERROR("MNIST Torch classification failed: {}", exception.what());
        return 1;
    }
}
