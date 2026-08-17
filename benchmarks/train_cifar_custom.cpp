#include "experiment_config.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/ClassificationLoader.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Profiler.hpp"
#include "SimpleCNN.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
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

auto batch_accuracy(const dl::Tensor& logits, const dl::Tensor& one_hot, cudaStream_t stream = 0) -> float
{
    const std::vector<float> logit_host = logits.to_host(stream);
    const std::vector<float> target_host = one_hot.to_host(stream);
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

auto for_each_prefetched_batch(ClassificationLoader& loader,
    const std::function<void(Batch&, int, cudaStream_t)>& step) -> int
{
    loader.reset();
    dl::UniqueCudaStream streams[2];
    std::optional<Batch> batches[2];
    bool ready[2] { false, false };
    if (loader.has_next())
    {
        batches[0] = loader.get_batch(streams[0].get());
        ready[0] = true;
    }

    int slot = 0;
    int count = 0;
    while (ready[slot])
    {
        const int next = 1 - slot;
        CHECK_CUDA(cudaStreamSynchronize(streams[slot].get()));
        const dl::StreamGuard stream_guard(streams[slot].get());
        step(*batches[slot], count, streams[slot].get());
        ++count;
        if (loader.has_next())
        {
            CHECK_CUDA(cudaStreamSynchronize(streams[next].get()));
            batches[next] = loader.get_batch(streams[next].get());
            ready[next] = true;
        }
        else
        {
            ready[next] = false;
            batches[next].reset();
        }
        slot = next;
    }
    CHECK_CUDA(cudaStreamSynchronize(streams[0].get()));
    CHECK_CUDA(cudaStreamSynchronize(streams[1].get()));
    return count;
}

int main()
{
    try
    {
        const nlohmann::json config = load_pipeline_config("cifar10_classification");
        apply_pipeline_precision(config);
        const int batch_size = config.value("batch_size", 64);
        const int total_epochs = config.value("epochs", 20);
        const float learning_rate = config.value("learning_rate", 1.0e-3F);
        const float gradient_clip = pipeline_gradient_clip(config);
        const int image_size = config.value("image_size", 32);
        const std::string train_split = config.value("train_split", "train");
        const std::string test_split = config.value("test_split", "test");
        const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/cifar10"));
        const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/cifar10"));

        int gpu_count = 0;
        cudaGetDeviceCount(&gpu_count);
        LOG_INFO("[CIFAR-10 CLASSIFICATION] Starting on device: {}", gpu_count > 0 ? "GPU" : "CPU");
        LOG_INFO("[CONFIG] batch_size={} epochs={} learning_rate={} gradient_clip={} dataset_root={}", batch_size,
            total_epochs, learning_rate, gradient_clip, data_root.string());
        LOG_FLUSH();

        LOG_INFO("Scanning train split '{}' ...", train_split);
        LOG_FLUSH();
        ClassificationLoader train_loader(data_root.string(), train_split, batch_size, image_size, true);
        LOG_INFO("Scanning test split '{}' ...", test_split);
        LOG_FLUSH();
        const std::vector<std::string> class_names = train_loader.class_names();
        ClassificationLoader test_loader(
            data_root.string(), test_split, batch_size, image_size, false, class_names);
        const int num_classes = train_loader.num_classes();
        if (test_loader.num_classes() != num_classes)
        {
            throw std::runtime_error("CIFAR train/test class counts differ: train=" + std::to_string(num_classes)
                + " test=" + std::to_string(test_loader.num_classes()));
        }
        LOG_INFO("[CONFIG] classes={} train={} test={}", num_classes, train_loader.size(), test_loader.size());
        if (train_loader.size() != 50000 || test_loader.size() != 10000)
        {
            LOG_WARN("CIFAR-10 expected 50000 train / 10000 test images; got {} / {}. Incomplete extract?",
                train_loader.size(), test_loader.size());
        }
        LOG_FLUSH();

        LOG_INFO("Constructing SimpleCNN ...");
        LOG_FLUSH();
        SimpleCNN model(num_classes, image_size);
        Network trainer(model.get_all_layers(), learning_rate, gradient_clip);
        LOG_INFO("Moving {} layers to GPU ...", model.get_all_layers().size());
        LOG_FLUSH();
        for (auto& layer : model.get_all_layers())
        {
            layer->to(dl::Device::GPU);
            layer->learning_rate = learning_rate;
            layer->train();
        }
        LOG_INFO("Model on GPU. Opening metrics at {}", (results_dir / "metrics_custom.csv").string());
        LOG_FLUSH();

        fs::create_directories(results_dir);
        std::ofstream csv_file((results_dir / "metrics_custom.csv").string());
        csv_file << "Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;TrainAcc;TestAcc\n";

        Profiler profiler;
        for (int epoch = 1; epoch <= total_epochs; ++epoch)
        {
            auto epoch_start = std::chrono::steady_clock::now();
            profiler.start();
            LOG_INFO("CIFAR-10 epoch {}/{} train start ({} images, batch {})", epoch, total_epochs, train_loader.size(),
                batch_size);
            LOG_FLUSH();

            for (auto& layer : model.get_all_layers())
            {
                layer->train();
                layer->learning_rate = learning_rate;
            }

            float train_loss = 0.0F;
            float train_acc = 0.0F;
            const int train_batches = for_each_prefetched_batch(train_loader,
                [&](Batch& batch, int index, cudaStream_t stream)
                {
                    if (index == 0)
                    {
                        LOG_INFO("First batch images {} targets {}", batch.images.describe(), batch.targets.describe());
                        LOG_INFO("Running first forward (cuDNN algo pick happens here) ...");
                        LOG_FLUSH();
                    }
                    dl::Tensor logits = model.forward_logits(batch.images, stream);
                    if (index == 0)
                    {
                        LOG_INFO("First logits {} vs targets {}", logits.describe(), batch.targets.describe());
                        LOG_FLUSH();
                    }
                    train_loss += CrossEntropyLoss::loss(batch.targets, logits).to_host(stream).front();
                    train_acc += batch_accuracy(logits, batch.targets, stream);

                    dl::Tensor grad =
                        trainer.clip_loss_gradient(CrossEntropyLoss::loss_derivative(batch.targets, logits));
                    auto layers = model.get_all_layers();
                    for (auto iterator = layers.rbegin(); iterator != layers.rend(); ++iterator)
                    {
                        grad = (*iterator)->backward(grad, stream);
                    }
                    trainer.clip_parameter_gradients(stream);
                    for (auto& layer : layers)
                    {
                        layer->step(stream);
                    }
                    if (index == 0 || (index + 1) % 50 == 0)
                    {
                        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - epoch_start)
                                                 .count();
                        const int done = index + 1;
                        LOG_INFO("CIFAR-10 train epoch {} batch {} last_loss={:.4f} elapsed={}s", epoch, done,
                            train_loss / static_cast<float>(done), elapsed);
                        LOG_FLUSH();
                    }
                });

            LOG_INFO("CIFAR-10 epoch {} train done ({} batches). Starting eval ...", epoch, train_batches);
            LOG_FLUSH();
            for (auto& layer : model.get_all_layers())
            {
                layer->eval();
            }

            float test_loss = 0.0F;
            float test_acc = 0.0F;
            const int test_batches = for_each_prefetched_batch(test_loader,
                [&](Batch& batch, int index, cudaStream_t stream)
                {
                    dl::Tensor logits = model.forward_logits(batch.images, stream);
                    test_loss += CrossEntropyLoss::loss(batch.targets, logits).to_host(stream).front();
                    test_acc += batch_accuracy(logits, batch.targets, stream);
                    if (index == 0 || (index + 1) % 50 == 0)
                    {
                        LOG_INFO("CIFAR-10 eval epoch {} batch {}", epoch, index + 1);
                        LOG_FLUSH();
                    }
                });

            const float gpu_ms = profiler.stop();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - epoch_start)
                                     .count();
            const float avg_train = train_loss / static_cast<float>(std::max(1, train_batches));
            const float avg_test = test_loss / static_cast<float>(std::max(1, test_batches));
            const float avg_train_acc = train_acc / static_cast<float>(std::max(1, train_batches));
            const float avg_test_acc = test_acc / static_cast<float>(std::max(1, test_batches));

            const auto vram = current_vram_mib();
            log_train_epoch("CIFAR-10 Custom", epoch, total_epochs, avg_train, avg_test, elapsed, vram);
            LOG_INFO("CIFAR-10 Custom | Train Acc: {:.4f} | Test Acc: {:.4f} | GPU: {} ms", avg_train_acc, avg_test_acc,
                gpu_ms);
            LOG_FLUSH();
            csv_file << epoch << ";" << avg_train << ";" << avg_test << ";" << elapsed << ";" << vram << ";"
                     << avg_train_acc << ";" << avg_test_acc << "\n";
            csv_file.flush();
        }

        const std::string save_path = (results_dir / "simplecnn_cifar10_final.bin").string();
        trainer.save(save_path);
        LOG_INFO("Final model saved: {}", save_path);
        LOG_FLUSH();
        return 0;
    }
    catch (const std::exception& exception)
    {
        LOG_ERROR("CIFAR-10 classification failed: {}", exception.what());
        LOG_FLUSH();
        return 1;
    }
}
