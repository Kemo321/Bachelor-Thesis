#include "classification_eval.hpp"
#include "classification_vis.hpp"
#include "experiment_config.hpp"
#include "run_metrics.hpp"

#include "DeepLearnLib/ClassificationLoader.hpp"
#include "DeepLearnLib/Logger.hpp"
#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/Network.hpp"
#include "DeepLearnLib/Profiler.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "SimpleCNN.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main()
{
    try
    {
        // Read the cifar10_classification pipeline JSON so the epoch count, learning rate, and paths come from the experiment.
        const nlohmann::json config = load_pipeline_config("cifar10_classification");
        const int batch_size = config.value("batch_size", 64);
        const int total_epochs = config.value("epochs", 20);
        const float learning_rate = config.value("learning_rate", 1.0e-3F);
        const float momentum = config.value("momentum", 0.9F);
        const float weight_decay = config.value("weight_decay", 0.0005F);
        const float gradient_clip = pipeline_gradient_clip(config);
        const int image_size = config.value("image_size", 32);
        const std::string train_split = config.value("train_split", "train");
        const std::string test_split = config.value("test_split", "test");
        const fs::path data_root = resolve_from_source(config.value("dataset_root", "data/cifar10"));
        const fs::path results_dir = resolve_from_source(config.value("results_dir", "results/cifar10"));
        // Keep the console log next to the CSV. A new run replaces this file.
        open_results_log(results_dir, "log_custom.txt");

        // Separate train and test splits so evaluation walks images held out of the training epoch.
        ClassificationLoader train_loader(data_root.string(), train_split, batch_size, image_size, true);
        const std::vector<std::string> class_names = train_loader.class_names();
        ClassificationLoader test_loader(
            data_root.string(), test_split, batch_size, image_size, false, class_names);
        const int num_classes = train_loader.num_classes();
        if (test_loader.num_classes() != num_classes)
        {
            throw std::runtime_error("CIFAR train/test class counts differ: train=" + std::to_string(num_classes)
                + " test=" + std::to_string(test_loader.num_classes()));
        }
        if (train_loader.size() != 50000 || test_loader.size() != 10000)
        {
            LOG_WARN("CIFAR-10 expected 50000 train / 10000 test images; got {} / {}. Incomplete extract?",
                train_loader.size(), test_loader.size());
        }

        // Build SimpleCNN and the SGD trainer on the GPU so the forward pass and the weight update stay on the device.
        SimpleCNN model(num_classes, image_size);
        Network trainer(model.get_all_layers(), learning_rate, gradient_clip);
        for (auto& layer : model.get_all_layers())
        {
            layer->to(dl::Device::GPU);
            layer->train();
        }
        apply_sgd_hyperparameters(model.get_all_layers(), learning_rate, momentum, weight_decay);
        log_pipeline_banner({ "CIFAR-10 Custom", "custom", batch_size, total_epochs, learning_rate, momentum,
            weight_decay, gradient_clip, num_classes, 0,
            static_cast<int>(model.get_all_layers().size()), data_root.string(),
            static_cast<std::size_t>(train_loader.size()), static_cast<std::size_t>(test_loader.size()) });
        LOG_FLUSH();

        fs::create_directories(results_dir);
        write_class_names(results_dir / "class_names.txt", class_names);
        // Metrics CSV so each epoch appends a row for comparing runs later.
        std::ofstream csv_file((results_dir / "metrics_custom.csv").string());
        csv_file << kClassificationCsvHeader << "\n";

        Profiler profiler;
        std::vector<int> confusion;
        std::vector<SamplePrediction> samples;
        for (int epoch = 1; epoch <= total_epochs; ++epoch)
        {
            auto epoch_start = std::chrono::steady_clock::now();
            profiler.start();

            const float current_lr = scheduled_learning_rate(config, epoch);
            apply_sgd_hyperparameters(model.get_all_layers(), current_lr, momentum, weight_decay);
            for (auto& layer : model.get_all_layers())
            {
                layer->train();
            }

            // Training epoch: cross-entropy, backpropagation, and an SGD step so the weights fit the training images.
            float train_loss = 0.0F;
            float train_acc = 0.0F;
            const int train_batches = for_each_prefetched_batch(train_loader,
                [&](Batch& batch, int index, cudaStream_t stream)
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
                    if (index == 0 || (index + 1) % 50 == 0)
                    {
                        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - epoch_start)
                                                 .count();
                        const int done = index + 1;
                        LOG_DEBUG("CIFAR-10 train epoch {} batch {} last_loss={:.4f} elapsed={}s", epoch, done,
                            train_loss / static_cast<float>(done), elapsed);
                    }
                });

            LOG_DEBUG("CIFAR-10 epoch {} train done ({} batches). Starting eval ...", epoch, train_batches);
            // Test pass in eval mode, to measure loss and accuracy on the test set.
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

            log_train_epoch({ "CIFAR-10 Custom", epoch, total_epochs, current_lr, avg_train, avg_test, avg_train_acc,
                avg_test_acc, std::nullopt, train_batches, elapsed, current_vram_mib() });
            LOG_DEBUG("CIFAR-10 Custom | GPU: {} ms", gpu_ms);
            write_classification_row(csv_file, epoch, avg_train, avg_test, elapsed, current_vram_mib(), avg_train_acc,
                avg_test_acc);
        }

        write_confusion_csv(results_dir / "confusion_custom.csv", confusion, num_classes, class_names);
        write_classification_samples(results_dir / "samples_custom", samples, class_names);
        // Save the weights at the end of the run so the same model can be loaded without repeating training.
        const std::string save_path = (results_dir / "simplecnn_cifar10_final.bin").string();
        trainer.save(save_path);
        log_saved("CIFAR-10 Custom", save_path);
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
