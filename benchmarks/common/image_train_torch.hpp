#pragma once

#include "classification_eval.hpp"
#include "classification_vis.hpp"
#include "experiment_config.hpp"
#include "torch_optim.hpp"
#include "train_loop.hpp"

#include <torch/torch.h>

#include <string>
#include <utility>
#include <vector>

inline auto batch_to_torch(const Batch& batch, const torch::Device& device)
    -> std::pair<torch::Tensor, torch::Tensor>
{
    const auto& image_shape = batch.images.get_shape();
    const auto& target_shape = batch.targets.get_shape();
    const int count = image_shape[0];
    const int classes = target_shape[1];
    std::vector<float> image_host = batch.images.to_host();
    std::vector<float> target_host = batch.targets.to_host();
    auto images = torch::from_blob(image_host.data(), { count, image_shape[1], image_shape[2], image_shape[3] },
        torch::kFloat32)
                      .clone()
                      .to(device);
    std::vector<int64_t> labels(static_cast<std::size_t>(count));
    for (int row = 0; row < count; ++row)
    {
        labels[static_cast<std::size_t>(row)] = classification_argmax_row(target_host, row, classes);
    }
    auto targets
        = torch::from_blob(labels.data(), { count }, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
    return { images, targets };
}

template <typename Loader, typename Module>
auto measure_torch_image_split(Loader& loader, Module& model, torch::nn::CrossEntropyLoss& criterion,
    torch::optim::Optimizer* optimizer, float gradient_clip, const torch::Device& device, int num_classes,
    std::vector<int>* confusion, std::vector<SamplePrediction>* samples) -> SplitMetrics
{
    SplitMetrics stats;
    int seen = 0;
    loader.reset();
    while (loader.has_next())
    {
        Batch batch = loader.get_batch();
        auto tensors = batch_to_torch(batch, device);
        torch::Tensor logits;
        float loss_value = 0.0F;
        if (optimizer != nullptr)
        {
            optimizer->zero_grad();
            logits = model->forward(tensors.first);
            auto loss = criterion(logits, tensors.second);
            loss.backward();
            clip_torch_grad_value(*model, gradient_clip);
            optimizer->step();
            loss_value = loss.item<float>();
        }
        else
        {
            torch::NoGradGuard no_grad;
            logits = model->forward(tensors.first);
            loss_value = criterion(logits, tensors.second).item<float>();
        }

        const int count = batch.images.get_shape()[0];
        const auto predicted = logits.argmax(1);
        const int correct = static_cast<int>(predicted.eq(tensors.second).sum().item<int64_t>());
        stats.observe(loss_value, correct, count);
        if (confusion != nullptr || samples != nullptr)
        {
            const auto pred_cpu = predicted.to(torch::kCPU);
            const auto truth_cpu = tensors.second.to(torch::kCPU);
            const auto prob_cpu = torch::softmax(logits, 1).to(torch::kCPU);
            std::vector<int> truths(static_cast<std::size_t>(count));
            std::vector<int> preds(static_cast<std::size_t>(count));
            std::vector<float> confidences(static_cast<std::size_t>(count));
            for (int row = 0; row < count; ++row)
            {
                truths[static_cast<std::size_t>(row)] = static_cast<int>(truth_cpu[row].item<int64_t>());
                preds[static_cast<std::size_t>(row)] = static_cast<int>(pred_cpu[row].item<int64_t>());
                confidences[static_cast<std::size_t>(row)]
                    = prob_cpu[row][preds[static_cast<std::size_t>(row)]].item<float>();
            }
            if (confusion != nullptr)
            {
                accumulate_confusion_ids(*confusion, truths, preds, num_classes);
            }
            if (samples != nullptr)
            {
                append_samples_from_ids(batch.images, truths, preds, confidences, seen, 24, *samples);
            }
        }
        seen += count;
    }
    return stats;
}

template <typename TrainLoader, typename TestLoader, typename Module>
auto train_torch_classifier(const nlohmann::json& config, TrainLoader& train_loader, TestLoader& test_loader,
    Module& model, const torch::Device& device, const std::filesystem::path& results_dir,
    const std::string& weight_filename, const std::vector<std::string>& class_names) -> void
{
    const OptimFields opt = read_optim_fields(config, 64, 20, 1.0e-3F, 0.9F, 0.0005F);
    prepare_torch_experiment();
    const int num_classes = train_loader.num_classes();
    if (test_loader.num_classes() != num_classes)
    {
        throw std::runtime_error("Train and test class counts differ");
    }

    model->to(device);
    torch::optim::SGD optimizer = make_sgd(*model, scheduled_learning_rate(config, 1), opt.momentum, opt.weight_decay);
    torch::nn::CrossEntropyLoss criterion;

    write_class_names(results_dir / "class_names.txt", class_names);
    std::ofstream csv = open_metrics_csv(results_dir, "metrics_torch.csv", kMetricsCsvHeader);
    std::vector<int> confusion;
    std::vector<SamplePrediction> samples;

    run_supervised_epochs(
        opt.epochs, csv,
        [&](int epoch)
        {
            const float learning_rate = scheduled_learning_rate(config, epoch);
            set_sgd_lr(optimizer, learning_rate);
            model->train();
            return learning_rate;
        },
        [&](int)
        {
            return measure_torch_image_split(train_loader, model, criterion, &optimizer, opt.gradient_clip, device,
                num_classes, nullptr, nullptr);
        },
        [&](int epoch)
        {
            model->eval();
            const bool last = epoch == opt.epochs;
            if (last)
            {
                confusion.assign(static_cast<std::size_t>(num_classes) * static_cast<std::size_t>(num_classes), 0);
                samples.clear();
            }
            return measure_torch_image_split(test_loader, model, criterion, nullptr, opt.gradient_clip, device,
                num_classes, last ? &confusion : nullptr, last ? &samples : nullptr);
        });

    write_confusion_csv(results_dir / "confusion_torch.csv", confusion, num_classes, class_names);
    write_classification_samples(results_dir / "samples_torch", samples, class_names);
    torch::save(model, (results_dir / weight_filename).string());
}
