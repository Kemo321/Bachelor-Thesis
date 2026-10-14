#pragma once

#include "classification_eval.hpp"
#include "classification_vis.hpp"
#include "custom_step.hpp"
#include "experiment_config.hpp"
#include "train_loop.hpp"

#include "DeepLearnLib/core/Network.hpp"
#include "DeepLearnLib/losses/Losses.hpp"

#include <string>
#include <vector>

/**
 * Image classification epoch used by CIFAR, MNIST, and the fused CIFAR run.
 * The Torch twin is train_torch_classifier: same schedule, same sample-weighted metrics, same artefacts.
 */
template <typename Loader, typename Model>
auto measure_custom_image_split(Loader& loader, Model& model, const std::vector<std::shared_ptr<Layer>>& layers,
    bool update, int num_classes, std::vector<int>* confusion, std::vector<SamplePrediction>* samples) -> SplitMetrics
{
    SplitMetrics stats;
    int seen = 0;
    for_each_prefetched_batch(loader, [&](Batch& batch, int, cudaStream_t stream)
        {
        dl::Tensor logits = model.forward_logits(batch.images, stream);
        const float loss = CrossEntropyLoss::loss(batch.targets, logits).to_host(stream).front();
        const int count = batch.images.get_shape()[0];
        const int correct = count_correct_one_hot(logits, batch.targets, stream);
        stats.observe(loss, correct, count);
        if (confusion != nullptr)
        {
            accumulate_confusion_one_hot(*confusion, logits, batch.targets, num_classes, stream);
        }
        if (samples != nullptr)
        {
            collect_batch_predictions(batch.images, logits, batch.targets, seen, 24, *samples, stream);
        }
        seen += count;
        if (update)
        {
            backward_and_step(layers, CrossEntropyLoss::loss_derivative(batch.targets, logits), stream);
        } });
    return stats;
}

template <typename TrainLoader, typename TestLoader, typename Model>
auto train_custom_classifier(const nlohmann::json& config, TrainLoader& train_loader, TestLoader& test_loader,
    Model& model, const std::filesystem::path& results_dir, const std::string& weight_filename,
    const std::vector<std::string>& class_names) -> void
{
    const OptimFields opt = read_optim_fields(config, 64, 20, 1.0e-3F, 0.9F, 0.0005F);
    const int num_classes = train_loader.num_classes();
    if (test_loader.num_classes() != num_classes)
    {
        throw std::runtime_error("Train and test class counts differ");
    }

    auto layers = model.get_all_layers();
    Network checkpoint(layers, opt.learning_rate, opt.gradient_clip);
    configure_custom_layers(layers, opt.learning_rate, opt.momentum, opt.weight_decay, opt.gradient_clip, false);

    write_class_names(results_dir / "class_names.txt", class_names);
    std::ofstream csv = open_metrics_csv(results_dir, "metrics_custom.csv", kMetricsCsvHeader);
    std::vector<int> confusion;
    std::vector<SamplePrediction> samples;

    run_supervised_epochs(
        opt.epochs, csv,
        [&](int epoch)
        {
            const float learning_rate = scheduled_learning_rate(config, epoch);
            apply_sgd_hyperparameters(layers, learning_rate, opt.momentum, opt.weight_decay);
            for (const auto& layer : layers)
            {
                layer->train();
            }
            return learning_rate;
        },
        [&](int)
        {
            return measure_custom_image_split(train_loader, model, layers, true, num_classes, nullptr, nullptr);
        },
        [&](int epoch)
        {
            for (const auto& layer : layers)
            {
                layer->eval();
            }
            const bool last = epoch == opt.epochs;
            if (last)
            {
                confusion.assign(static_cast<std::size_t>(num_classes) * static_cast<std::size_t>(num_classes), 0);
                samples.clear();
            }
            return measure_custom_image_split(test_loader, model, layers, false, num_classes,
                last ? &confusion : nullptr, last ? &samples : nullptr);
        });

    write_confusion_csv(results_dir / "confusion_custom.csv", confusion, num_classes, class_names);
    write_classification_samples(results_dir / "samples_custom", samples, class_names);
    checkpoint.save((results_dir / weight_filename).string());
}
