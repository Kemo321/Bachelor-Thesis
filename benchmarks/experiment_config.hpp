#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Loads config/experiments.json for reproducible pipeline hyperparameters.
 *
 * Search order: EXPERIMENTS_JSON, DEEPLEARN_SOURCE_DIR/config/experiments.json,
 * then a few paths relative to the current working directory.
 */
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

/**
 * @brief Piecewise-constant LR from config["lr_schedule"] (ordered by until_epoch).
 *
 * Falls back to config["learning_rate"] when the schedule is absent or empty.
 */
inline auto scheduled_learning_rate(const nlohmann::json& config, int epoch) -> float
{
    const float fallback = config.value("learning_rate", 1.0e-4F);
    if (!config.contains("lr_schedule") || !config.at("lr_schedule").is_array() || config.at("lr_schedule").empty())
    {
        return fallback;
    }

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
