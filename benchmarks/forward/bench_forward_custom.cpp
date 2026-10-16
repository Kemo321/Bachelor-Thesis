#include "common/custom_step.hpp"
#include "common/forward_plan.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "models/GruClassifier.hpp"
#include "models/SimpleCNN.hpp"
#include "models/TabularMLP.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace
{

auto constant_tensor(const std::vector<int>& shape) -> dl::Tensor
{
    int count = 1;
    for (const int dim : shape)
    {
        count *= dim;
    }
    return dl::Tensor::from_host(shape, std::vector<float>(static_cast<std::size_t>(count), 0.5F), dl::Device::GPU);
}

template <typename Model>
auto prepare_eval(Model& model) -> void
{
    for (const auto& layer : model.get_all_layers())
    {
        layer->to(dl::Device::GPU);
        layer->eval();
    }
}

template <typename Fn>
auto time_gpu_calls(int warmup, int repeats, Fn step) -> double
{
    for (int index = 0; index < warmup; ++index)
    {
        step();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto started = std::chrono::steady_clock::now();
    for (int index = 0; index < repeats; ++index)
    {
        step();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return elapsed_seconds(started);
}

}

int main()
{
    try
    {
        const ForwardPlan plan = load_forward_plan();
        std::ofstream csv = open_metrics_csv(plan.results_dir, "metrics_custom.csv", kForwardCsvHeader);
        for (const ForwardCase& item : plan.cases)
        {
            double seconds = 0.0;
            if (item.kind == "simplecnn")
            {
                SimpleCNN model(item.num_classes, item.image_size, item.in_channels);
                prepare_eval(model);
                dl::Tensor input = constant_tensor({ item.batch, item.in_channels, item.image_size, item.image_size });
                seconds = time_gpu_calls(plan.warmup, item.repeats, [&]
                    { (void)model.forward_logits(input); });
            }
            else if (item.kind == "tabular")
            {
                TabularMLP model(item.num_features, item.hidden_size, item.num_classes);
                prepare_eval(model);
                dl::Tensor input = constant_tensor({ item.batch, item.num_features });
                seconds = time_gpu_calls(plan.warmup, item.repeats, [&]
                    { (void)model.forward_logits(input); });
            }
            else if (item.kind == "gru")
            {
                GruClassifier model(item.input_size, item.hidden_size, item.num_classes);
                prepare_eval(model);
                dl::Tensor input = constant_tensor({ item.batch, item.length, item.input_size });
                seconds = time_gpu_calls(plan.warmup, item.repeats, [&]
                    { (void)model.forward_logits(input); });
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
        std::fprintf(stderr, "Forward benchmark failed: %s\n", exception.what());
        return 1;
    }
}
