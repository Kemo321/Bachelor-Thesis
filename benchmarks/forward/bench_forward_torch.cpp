#include "common/forward_plan.hpp"
#include "common/torch_optim.hpp"
#include "models/torch/TorchGruClassifier.hpp"
#include "models/torch/TorchSimpleCNN.hpp"
#include "models/torch/TorchTabularMLP.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace
{

template <typename Fn>
auto time_torch_calls(int warmup, int repeats, Fn step) -> double
{
    for (int index = 0; index < warmup; ++index)
    {
        step();
    }
    sync_torch_device();
    const auto started = std::chrono::steady_clock::now();
    for (int index = 0; index < repeats; ++index)
    {
        step();
    }
    sync_torch_device();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

}

int main()
{
    try
    {
        const ForwardPlan plan = load_forward_plan();
        prepare_torch_experiment();
        const torch::Device device = torch_device();
        std::ofstream csv = open_metrics_csv(plan.results_dir, "metrics_torch.csv", kForwardCsvHeader);
        for (const ForwardCase& item : plan.cases)
        {
            double seconds = 0.0;
            torch::NoGradGuard no_grad;
            if (item.kind == "simplecnn")
            {
                TorchSimpleCNN model(item.num_classes, item.image_size, item.in_channels);
                model->to(device);
                model->eval();
                auto input = torch::full({ item.batch, item.in_channels, item.image_size, item.image_size }, 0.5,
                    torch::TensorOptions().device(device));
                seconds = time_torch_calls(plan.warmup, item.repeats, [&]
                    { (void)model->forward(input); });
            }
            else if (item.kind == "tabular")
            {
                TorchTabularMLP model(item.num_features, item.hidden_size, item.num_classes);
                model->to(device);
                model->eval();
                auto input = torch::full({ item.batch, item.num_features }, 0.5, torch::TensorOptions().device(device));
                seconds = time_torch_calls(plan.warmup, item.repeats, [&]
                    { (void)model->forward(input); });
            }
            else if (item.kind == "gru")
            {
                TorchGruClassifier model(item.input_size, item.hidden_size, item.num_classes);
                model->to(device);
                model->eval();
                auto input = torch::full({ item.batch, item.length, item.input_size }, 0.5,
                    torch::TensorOptions().device(device));
                seconds = time_torch_calls(plan.warmup, item.repeats, [&]
                    { (void)model->forward(input); });
            }
            else
            {
                throw std::runtime_error("Unknown forward case kind: " + item.kind);
            }
            write_forward_row(csv, item.name, item.batch, item.repeats, seconds, current_vram_mib());
        }
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "Torch forward benchmark failed: %s\n", exception.what());
        return 1;
    }
}
