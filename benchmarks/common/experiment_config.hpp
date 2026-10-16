#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "DeepLearnLib/core/Layer.hpp"

inline auto experiments_json_path() -> std::filesystem::path
{
    if (const char* from_env = std::getenv("EXPERIMENTS_JSON"))
    {
        std::filesystem::path env_path(from_env);
        if (std::filesystem::exists(env_path))
        {
            return env_path;
        }
    }

    std::vector<std::filesystem::path> candidates;
#ifdef DEEPLEARN_SOURCE_DIR
    candidates.emplace_back(std::filesystem::path(DEEPLEARN_SOURCE_DIR) / "config" / "experiments.json");
#endif
    candidates.emplace_back(std::filesystem::current_path() / "config" / "experiments.json");
    candidates.emplace_back(std::filesystem::current_path() / ".." / "config" / "experiments.json");
    candidates.emplace_back(std::filesystem::current_path() / ".." / ".." / "config" / "experiments.json");

    for (const auto& candidate : candidates)
    {
        std::error_code error;
        if (std::filesystem::exists(candidate, error))
        {
            return std::filesystem::weakly_canonical(candidate, error);
        }
    }
    throw std::runtime_error("Could not find config/experiments.json (set EXPERIMENTS_JSON)");
}

inline auto load_experiments_json() -> nlohmann::json
{
    const auto path = experiments_json_path();
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("Failed to open experiments config: " + path.string());
    }
    nlohmann::json document;
    stream >> document;
    return document;
}

inline auto load_pipeline_config(const std::string& pipeline_name) -> nlohmann::json
{
    const nlohmann::json document = load_experiments_json();
    if (!document.contains(pipeline_name))
    {
        throw std::runtime_error("experiments.json has no pipeline named '" + pipeline_name + "'");
    }
    return document.at(pipeline_name);
}

inline auto resolve_from_source(const std::string& maybe_relative) -> std::filesystem::path
{
    std::filesystem::path path(maybe_relative);
    if (path.is_absolute())
    {
        return path;
    }
#ifdef DEEPLEARN_SOURCE_DIR
    return std::filesystem::path(DEEPLEARN_SOURCE_DIR) / path;
#else
    return std::filesystem::current_path() / path;
#endif
}

inline auto pipeline_gradient_clip(const nlohmann::json& config) -> float
{
    return config.value("gradient_clip", 0.0F);
}

struct OptimFields
{
    int batch_size = 1;
    int epochs = 1;
    float learning_rate = 1.0e-3F;
    float momentum = 0.9F;
    float weight_decay = 0.0F;
    float gradient_clip = 0.0F;
    float test_fraction = 0.2F;
};

inline auto read_optim_fields(const nlohmann::json& config, int default_batch, int default_epochs, float default_lr,
    float default_momentum, float default_weight_decay) -> OptimFields
{
    OptimFields fields;
    fields.batch_size = config.value("batch_size", default_batch);
    fields.epochs = config.value("epochs", default_epochs);
    fields.learning_rate = config.value("learning_rate", default_lr);
    fields.momentum = config.value("momentum", default_momentum);
    fields.weight_decay = config.value("weight_decay", default_weight_decay);
    fields.gradient_clip = pipeline_gradient_clip(config);
    fields.test_fraction = config.value("test_fraction", 0.2F);
    return fields;
}

struct NamedPipeline
{
    nlohmann::json config;
    std::filesystem::path data_root;
    std::filesystem::path results_dir;
};

inline auto open_pipeline(const std::string& pipeline_name, const std::string& default_root,
    const std::string& default_results) -> NamedPipeline
{
    nlohmann::json config = load_pipeline_config(pipeline_name);
    NamedPipeline pipeline;
    pipeline.config = std::move(config);
    pipeline.data_root = resolve_from_source(pipeline.config.value("dataset_root", default_root));
    pipeline.results_dir = resolve_from_source(pipeline.config.value("results_dir", default_results));
    return pipeline;
}

inline auto pipeline_name_from_args(int argc, char** argv, const std::string& fallback) -> std::string
{
    if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0')
    {
        return argv[1];
    }
    return fallback;
}

inline auto class_names_from_config(const nlohmann::json& config, int num_classes) -> std::vector<std::string>
{
    if (config.contains("class_names") && config.at("class_names").is_array()
        && static_cast<int>(config.at("class_names").size()) == num_classes)
    {
        return config.at("class_names").get<std::vector<std::string>>();
    }
    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(num_classes));
    for (int class_id = 0; class_id < num_classes; ++class_id)
    {
        names.push_back(std::to_string(class_id));
    }
    return names;
}

inline auto scheduled_learning_rate(const nlohmann::json& config, int epoch) -> float
{
    const float fallback = config.value("learning_rate", 1.0e-4F);
    if (config.contains("lr_schedule") && config.at("lr_schedule").is_array() && !config.at("lr_schedule").empty())
    {
        float last = fallback;
        for (const auto& step : config.at("lr_schedule"))
        {
            last = step.value("learning_rate", last);
            if (epoch <= step.value("until_epoch", epoch))
            {
                return last;
            }
        }
        return last;
    }

    const int total = std::max(1, config.value("epochs", epoch));
    const int drop_70 = std::max(1, static_cast<int>(std::lround(0.70 * static_cast<double>(total))));
    const int drop_90 = std::max(drop_70, static_cast<int>(std::lround(0.90 * static_cast<double>(total))));
    if (epoch <= drop_70)
    {
        return fallback;
    }
    if (epoch <= drop_90)
    {
        return fallback * 0.1F;
    }
    return fallback * 0.01F;
}

inline auto apply_sgd_hyperparameters(const std::vector<std::shared_ptr<Layer>>& layers, float learning_rate,
    float momentum, float weight_decay) -> void
{
    for (const auto& layer : layers)
    {
        layer->learning_rate = learning_rate;
        layer->momentum = momentum;
        layer->weight_decay = weight_decay;
    }
}
