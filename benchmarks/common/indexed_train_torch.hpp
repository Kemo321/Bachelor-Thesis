#pragma once

#include "classification_eval.hpp"
#include "experiment_config.hpp"
#include "row_batch.hpp"
#include "torch_optim.hpp"
#include "train_loop.hpp"

#include "DeepLearnLib/utils/ExperimentSeed.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

template <typename Module>
auto measure_torch_rows(const HostRows& table, const std::vector<int>& order, int batch_size, int num_classes,
    bool update, Module& model, torch::optim::Optimizer* optimizer, float gradient_clip, const torch::Device& device,
    std::vector<int>* confusion) -> SplitMetrics
{
    require_host_rows(table);
    SplitMetrics stats;
    const int available = static_cast<int>(order.size());
    const int step = std::max(1, std::min(batch_size, std::max(available, 1)));
    torch::nn::CrossEntropyLoss criterion;
    for (int start = 0; start < available; start += step)
    {
        const int count = std::min(step, available - start);
        const RowBatch host = gather_host_rows(table, order, start, count);
        std::vector<int64_t> label_ids(static_cast<std::size_t>(count));
        for (int row = 0; row < count; ++row)
        {
            label_ids[static_cast<std::size_t>(row)] = std::clamp(
                static_cast<int>(std::lround(host.labels[static_cast<std::size_t>(row)])), 0, num_classes - 1);
        }
        const std::vector<int64_t> shape = batch_shape_i64(count, table.tail);
        auto features = torch::from_blob(const_cast<float*>(host.values.data()), shape,
            torch::TensorOptions().dtype(torch::kFloat32))
                            .clone()
                            .to(device);
        auto targets = torch::from_blob(label_ids.data(), { count }, torch::TensorOptions().dtype(torch::kLong))
                           .clone()
                           .to(device);

        torch::Tensor logits;
        float loss_value = 0.0F;
        if (update && optimizer != nullptr)
        {
            optimizer->zero_grad();
            logits = model->forward(features);
            auto loss = criterion(logits, targets);
            loss.backward();
            clip_torch_grad_value(*model, gradient_clip);
            optimizer->step();
            loss_value = loss.item<float>();
        }
        else
        {
            torch::NoGradGuard no_grad;
            logits = model->forward(features);
            loss_value = criterion(logits, targets).item<float>();
        }

        const auto predicted = logits.argmax(1);
        const int correct = static_cast<int>(predicted.eq(targets).sum().item<int64_t>());
        stats.observe(loss_value, correct, count);
        if (confusion != nullptr)
        {
            const auto pred_cpu = predicted.to(torch::kCPU);
            std::vector<int> truths(static_cast<std::size_t>(count));
            std::vector<int> preds(static_cast<std::size_t>(count));
            for (int row = 0; row < count; ++row)
            {
                truths[static_cast<std::size_t>(row)] = static_cast<int>(label_ids[static_cast<std::size_t>(row)]);
                preds[static_cast<std::size_t>(row)] = static_cast<int>(pred_cpu[row].item<int64_t>());
            }
            accumulate_confusion_ids(*confusion, truths, preds, num_classes);
        }
    }
    return stats;
}

/** Same schedule, shuffle seed, split, and CSV columns as train_custom_rows. */
template <typename Module>
auto train_torch_rows(const nlohmann::json& config, const HostRows& train_rows, const HostRows& test_rows,
    std::vector<int> train_order, std::vector<int> test_order, Module& model, bool adam,
    const torch::Device& device, const std::filesystem::path& results_dir, const std::string& weight_filename,
    const std::vector<std::string>& class_names, int num_classes, int default_batch, int default_epochs,
    float default_lr, float default_weight_decay) -> void
{
    const OptimFields opt
        = read_optim_fields(config, default_batch, default_epochs, default_lr, 0.9F, default_weight_decay);
    prepare_torch_experiment();
    if (train_order.empty() || test_order.empty())
    {
        throw std::runtime_error("Row training needs a non-empty train split and a non-empty test split");
    }

    model->to(device);
    const float initial_lr = scheduled_learning_rate(config, 1);
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    if (adam)
    {
        optimizer = std::make_unique<torch::optim::Adam>(make_adam(*model, initial_lr, opt.weight_decay));
    }
    else
    {
        optimizer = std::make_unique<torch::optim::SGD>(make_sgd(*model, initial_lr, opt.momentum, opt.weight_decay));
    }

    write_class_names(results_dir / "class_names.txt", class_names);
    std::ofstream csv = open_metrics_csv(results_dir, "metrics_torch.csv", kMetricsCsvHeader);
    std::mt19937 rng(dl::kExperimentSeed);
    std::vector<int> confusion;

    run_supervised_epochs(
        opt.epochs, csv,
        [&](int epoch)
        {
            const float learning_rate = scheduled_learning_rate(config, epoch);
            if (adam)
            {
                set_adam_lr(*optimizer, learning_rate);
            }
            else
            {
                set_sgd_lr(*optimizer, learning_rate);
            }
            model->train();
            dl::portable_shuffle(train_order, rng);
            return learning_rate;
        },
        [&](int)
        {
            return measure_torch_rows(train_rows, train_order, opt.batch_size, num_classes, true, model, optimizer.get(),
                opt.gradient_clip, device, nullptr);
        },
        [&](int epoch)
        {
            model->eval();
            const bool last = epoch == opt.epochs;
            if (last)
            {
                confusion.assign(static_cast<std::size_t>(num_classes) * static_cast<std::size_t>(num_classes), 0);
            }
            return measure_torch_rows(test_rows, test_order, opt.batch_size, num_classes, false, model, nullptr,
                opt.gradient_clip, device, last ? &confusion : nullptr);
        });

    write_confusion_csv(results_dir / "confusion_torch.csv", confusion, num_classes, class_names);
    torch::save(model, (results_dir / weight_filename).string());
}
