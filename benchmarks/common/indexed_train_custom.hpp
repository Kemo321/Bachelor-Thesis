#pragma once

#include "classification_eval.hpp"
#include "custom_step.hpp"
#include "experiment_config.hpp"
#include "row_batch.hpp"
#include "tabular_common.hpp"
#include "train_loop.hpp"

#include "DeepLearnLib/core/Network.hpp"
#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/losses/Losses.hpp"
#include "DeepLearnLib/utils/ExperimentSeed.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

template <typename Forward>
auto measure_custom_rows(const HostRows& table, const std::vector<int>& order, int batch_size, int num_classes,
    bool update, const std::vector<std::shared_ptr<Layer>>& layers, Forward forward, std::vector<int>* confusion)
    -> SplitMetrics
{
    require_host_rows(table);
    SplitMetrics stats;
    const int available = static_cast<int>(order.size());
    const int step = std::max(1, std::min(batch_size, std::max(available, 1)));
    for (int start = 0; start < available; start += step)
    {
        const int count = std::min(step, available - start);
        const RowBatch host = gather_host_rows(table, order, start, count);
        dl::Tensor features = dl::Tensor::from_host(batch_shape(count, table.tail), host.values, dl::Device::GPU);
        dl::Tensor targets = dl::Tensor::from_host({ count, num_classes }, one_hot_labels(host.labels, num_classes),
            dl::Device::GPU);
        dl::Tensor logits = forward(features);
        const float loss = CrossEntropyLoss::loss(targets, logits).to_host().front();
        const std::vector<float> logit_host = logits.to_host();
        const int correct = count_correct_labels(logit_host, host.labels, num_classes);
        stats.observe(loss, correct, count);
        if (confusion != nullptr)
        {
            std::vector<int> truths(static_cast<std::size_t>(count));
            std::vector<int> preds(static_cast<std::size_t>(count));
            for (int row = 0; row < count; ++row)
            {
                truths[static_cast<std::size_t>(row)] = std::clamp(
                    static_cast<int>(std::lround(host.labels[static_cast<std::size_t>(row)])), 0, num_classes - 1);
                preds[static_cast<std::size_t>(row)] = classification_argmax_row(logit_host, row, num_classes);
            }
            accumulate_confusion_ids(*confusion, truths, preds, num_classes);
        }
        if (update)
        {
            backward_and_step(layers, CrossEntropyLoss::loss_derivative(targets, logits), 0);
        }
    }
    return stats;
}

/**
 * Row epoch used by the tabular MLP and the GRU classifier.
 * `train_order` is shuffled with seed 42 once per epoch. `test_order` stays fixed.
 * The Torch twin is train_torch_rows.
 */
template <typename Forward>
auto train_custom_rows(const nlohmann::json& config, const HostRows& train_rows, const HostRows& test_rows,
    std::vector<int> train_order, std::vector<int> test_order, Forward forward,
    const std::vector<std::shared_ptr<Layer>>& layers, bool adam, const std::filesystem::path& results_dir,
    const std::string& weight_filename, const std::vector<std::string>& class_names, int num_classes,
    int default_batch, int default_epochs, float default_lr, float default_weight_decay) -> void
{
    const OptimFields opt
        = read_optim_fields(config, default_batch, default_epochs, default_lr, 0.9F, default_weight_decay);
    if (train_order.empty() || test_order.empty())
    {
        throw std::runtime_error("Row training needs a non-empty train split and a non-empty test split");
    }

    Network checkpoint(layers, opt.learning_rate, opt.gradient_clip);
    configure_custom_layers(layers, opt.learning_rate, opt.momentum, opt.weight_decay, opt.gradient_clip, adam);

    write_class_names(results_dir / "class_names.txt", class_names);
    std::ofstream csv = open_metrics_csv(results_dir, "metrics_custom.csv", kMetricsCsvHeader);
    std::mt19937 rng(dl::kExperimentSeed);
    std::vector<int> confusion;

    run_supervised_epochs(
        opt.epochs, csv,
        [&](int epoch)
        {
            const float learning_rate = scheduled_learning_rate(config, epoch);
            apply_sgd_hyperparameters(layers, learning_rate, adam ? 0.0F : opt.momentum, opt.weight_decay);
            for (const auto& layer : layers)
            {
                layer->train();
            }
            dl::portable_shuffle(train_order, rng);
            return learning_rate;
        },
        [&](int)
        {
            return measure_custom_rows(train_rows, train_order, opt.batch_size, num_classes, true, layers, forward,
                nullptr);
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
            }
            return measure_custom_rows(test_rows, test_order, opt.batch_size, num_classes, false, layers, forward,
                last ? &confusion : nullptr);
        });

    write_confusion_csv(results_dir / "confusion_custom.csv", confusion, num_classes, class_names);
    checkpoint.save((results_dir / weight_filename).string());
}
