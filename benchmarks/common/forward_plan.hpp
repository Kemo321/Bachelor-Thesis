#pragma once

#include "experiment_config.hpp"
#include "run_metrics.hpp"

#include <string>
#include <vector>

/**
 * Forward-only latency cases. Custom and Torch read the same `forward_latency` block,
 * so the batch, the repeat count, and the CSV columns match.
 */
inline constexpr const char* kForwardCsvHeader = "Model;Batch;Repeats;Time(s);SamplesPerSec;VRAM_MiB";

struct ForwardCase
{
    std::string name;
    std::string kind;
    int batch = 1;
    int repeats = 1;
    int image_size = 32;
    int in_channels = 3;
    int num_classes = 10;
    int num_features = 4;
    int hidden_size = 16;
    int length = 96;
    int input_size = 1;
};

struct ForwardPlan
{
    std::filesystem::path results_dir;
    int warmup = 5;
    std::vector<ForwardCase> cases;
};

inline auto load_forward_plan() -> ForwardPlan
{
    const nlohmann::json config = load_pipeline_config("forward_latency");
    ForwardPlan plan;
    plan.results_dir = resolve_from_source(config.value("results_dir", "results/forward"));
    plan.warmup = config.value("warmup", 5);
    if (!config.contains("cases") || !config.at("cases").is_array() || config.at("cases").empty())
    {
        throw std::runtime_error("forward_latency.cases is empty");
    }
    for (const auto& item : config.at("cases"))
    {
        ForwardCase row;
        row.name = item.at("name").get<std::string>();
        row.kind = item.at("kind").get<std::string>();
        row.batch = item.value("batch", 1);
        row.repeats = item.value("repeats", 1);
        row.image_size = item.value("image_size", 32);
        row.in_channels = item.value("in_channels", 3);
        row.num_classes = item.value("num_classes", 10);
        row.num_features = item.value("num_features", 4);
        row.hidden_size = item.value("hidden_size", 16);
        row.length = item.value("length", 96);
        row.input_size = item.value("input_size", 1);
        if (row.batch < 1 || row.repeats < 1)
        {
            throw std::runtime_error("forward_latency case '" + row.name + "' needs a positive batch and repeat count");
        }
        plan.cases.push_back(std::move(row));
    }
    return plan;
}

inline auto write_forward_row(std::ofstream& csv, const std::string& name, int batch, int repeats, double time_s,
    std::size_t vram_mib) -> void
{
    const double samples = static_cast<double>(batch) * static_cast<double>(repeats);
    const double per_second = time_s > 0.0 ? samples / time_s : 0.0;
    csv << name << ";" << batch << ";" << repeats << ";" << format_decimal(time_s) << ";" << format_decimal(per_second)
        << ";" << vram_mib << "\n";
    csv.flush();
}
