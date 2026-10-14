#pragma once

#include "experiment_config.hpp"
#include "row_batch.hpp"
#include "tabular_common.hpp"

#include "DeepLearnLib/loaders/CSVLoader.hpp"

#include <string>
#include <vector>

struct TabularRun
{
    nlohmann::json config;
    std::filesystem::path results_dir;
    std::vector<float> features;
    std::vector<float> labels;
    int row_count = 0;
    int width = 0;
    int num_classes = 0;
    std::vector<std::string> class_names;
    IndexSplit split;
};

inline auto load_tabular_run(const std::string& pipeline_name) -> TabularRun
{
    const nlohmann::json config = load_pipeline_config(pipeline_name);
    const int num_classes = config.value("num_classes", 3);
    const int num_samples = config.value("num_samples", 64);
    const int num_features = config.value("num_features", 4);
    const bool skip_header = config.value("skip_header", true);
    const float test_fraction = config.value("test_fraction", 0.2F);
    const std::filesystem::path csv_path = resolve_from_source(config.value("csv_path", "data/tabular/demo.csv"));

    if (!std::filesystem::exists(csv_path))
    {
        write_dummy_csv(csv_path, num_samples, num_features, num_classes, dl::kExperimentSeed);
    }

    CSVLoader loader(csv_path.string(), 1, skip_header);
    TabularRun run;
    run.config = config;
    run.results_dir = resolve_from_source(config.value("results_dir", "results/tabular"));
    run.features = loader.features().to_host();
    run.labels = loader.targets().to_host();
    run.width = loader.features().get_shape()[1];
    run.row_count = static_cast<int>(loader.size());
    run.num_classes = num_classes;
    run.class_names = class_names_from_config(config, num_classes);
    run.split = split_train_test(run.row_count, test_fraction, dl::kExperimentSeed);
    return run;
}

inline auto tabular_host_rows(const TabularRun& run) -> HostRows
{
    HostRows rows;
    rows.values = &run.features;
    rows.labels = &run.labels;
    rows.rows = run.row_count;
    rows.width = run.width;
    rows.tail = { run.width };
    return rows;
}
