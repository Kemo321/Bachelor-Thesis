#include "classification_eval.hpp"
#include "classification_vis.hpp"
#include "experiment_config.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/PackedImageLoader.hpp"
#include "DeepLearnLib/Profiler.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "SimpleCNN.hpp"

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main()
{
    try
    {
        const nlohmann::json config = load_pipeline_config("mnist_classification");
        apply_pipeline_precision(config);
        const int batch_size = config.value("batch_size", 128);
        const int total_epochs = config.value("epochs", 10);
        const float learning_rate = config.value("learning_rate", 1.0e-2F);
        const float momentum = config.value("momentum", 0.9F);
        const float weight_decay = config.value("weight_decay", 0.0005F);
        const float gradient_clip = pipeline_gradient_clip(config);
        const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/mnist"));
        const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/mnist"));
        const fs::path train_bin = data_root / "train.bin";
        const fs::path test_bin = data_root / "test.bin";

        int gpu_count = 0;
        cudaGetDeviceCount(&gpu_count);
        LOG_INFO("[MNIST CLASSIFICATION] Starting on device: {}", gpu_count > 0 ? "GPU" : "CPU");
        LOG_INFO("[CONFIG] batch_size={} epochs={} learning_rate={} momentum={} weight_decay={} train_bin={}",
            batch_size, total_epochs, learning_rate, momentum, weight_decay, train_bin.string());

        PackedImageLoader train_loader(train_bin.string(), batch_size, true);
        PackedImageLoader test_loader(test_bin.string(), batch_size, false);
        const int num_classes = train_loader.num_classes();
        const int image_size = train_loader.height();
        const int in_channels = train_loader.channels();
        if (test_loader.num_classes() != num_classes || test_loader.channels() != in_channels)
        {
            throw std::runtime_error("MNIST train/test packed files disagree on shape or class count");
        }
        LOG_INFO("[CONFIG] classes={} channels={} {}x{} train={} test={}", num_classes, in_channels, image_size,
            train_loader.width(), train_loader.size(), test_loader.size());

        SimpleCNN model(num_classes, image_size, in_channels);
        Network trainer(model.get_all_layers(), learning_rate, gradient_clip);
        for (auto& layer : model.get_all_layers())
        {
            layer->to(dl::Device::GPU);
            layer->train();
        }
        apply_sgd_hyperparameters(model.get_all_layers(), learning_rate, momentum, weight_decay);

        fs::create_directories(results_dir);
        write_class_names(results_dir / "class_names.txt", train_loader.class_names());
        std::ofstream csv_file((results_dir / "metrics_custom.csv").string());
        csv_file << "Epoch;TrainLoss;TestLoss;Time(s);VRAM_MiB;TrainAcc;TestAcc\n";

        Profiler profiler;
        std::vector<int> confusion;
        std::vector<SamplePrediction> samples;
        for (int epoch = 1; epoch <= total_epochs; ++epoch)
        {
            auto epoch_start = std::chrono::steady_clock::now();
            profiler.start();
            apply_sgd_hyperparameters(model.get_all_layers(), scheduled_learning_rate(config, epoch), momentum,
                weight_decay);
            for (auto& layer : model.get_all_layers())
            {
                layer->train();
            }

            float train_loss = 0.0F;
            float train_acc = 0.0F;
            const int train_batches = for_each_prefetched_batch(train_loader,
                [&](Batch& batch, int, cudaStream_t stream)
                {
                    dl::Tensor logits = model.forward_logits(batch.images, stream);
                    train_loss += CrossEntropyLoss::loss(batch.targets, logits).to_host(stream).front();
                    train_acc += batch_accuracy_one_hot(logits, batch.targets, stream);
                    dl::Tensor grad = trainer.clip_loss_gradient(CrossEntropyLoss::loss_derivative(batch.targets, logits));
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
                });

            for (auto& layer : model.get_all_layers())
            {
                layer->eval();
            }

            float test_loss = 0.0F;
            float test_acc = 0.0F;
            confusion.assign(static_cast<std::size_t>(num_classes) * static_cast<std::size_t>(num_classes), 0);
            samples.clear();
            int seen_eval = 0;
            const int test_batches = for_each_prefetched_batch(test_loader,
                [&](Batch& batch, int, cudaStream_t stream)
                {
                    dl::Tensor logits = model.forward_logits(batch.images, stream);
                    test_loss += CrossEntropyLoss::loss(batch.targets, logits).to_host(stream).front();
                    test_acc += batch_accuracy_one_hot(logits, batch.targets, stream);
                    if (epoch == total_epochs)
                    {
                        accumulate_confusion_one_hot(confusion, logits, batch.targets, num_classes, stream);
                        collect_batch_predictions(batch.images, logits, batch.targets, seen_eval, 24, samples, stream);
                    }
                    seen_eval += batch.images.get_shape()[0];
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
            log_train_epoch("MNIST Custom", epoch, total_epochs, avg_train, avg_test, elapsed, vram);
            LOG_INFO("MNIST Custom | Train Acc: {:.4f} | Test Acc: {:.4f} | GPU: {} ms", avg_train_acc, avg_test_acc,
                gpu_ms);
            LOG_FLUSH();
            csv_file << epoch << ";" << avg_train << ";" << avg_test << ";" << elapsed << ";" << vram << ";"
                     << avg_train_acc << ";" << avg_test_acc << "\n";
            csv_file.flush();
        }

        write_confusion_csv(results_dir / "confusion_custom.csv", confusion, num_classes, train_loader.class_names());
        write_classification_samples(results_dir / "samples_custom", samples, train_loader.class_names());
        const std::string save_path = (results_dir / "simplecnn_mnist_final.bin").string();
        trainer.save(save_path);
        LOG_INFO("Final model saved: {}", save_path);
        LOG_FLUSH();
        return 0;
    }
    catch (const std::exception& exception)
    {
        LOG_ERROR("MNIST classification failed: {}", exception.what());
        LOG_FLUSH();
        return 1;
    }
}
