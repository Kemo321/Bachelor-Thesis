#include "common/indexed_train_torch.hpp"
#include "common/tabular_pipeline.hpp"
#include "models/torch/TorchTabularMLP.hpp"

#include <cstdio>

int main(int argc, char** argv)
{
    try
    {
        const std::string pipeline_name = pipeline_name_from_args(argc, argv, "tabular_demo");
        const TabularRun table = load_tabular_run(pipeline_name);
        const HostRows rows = tabular_host_rows(table);
        const int hidden_size = table.config.value("hidden_size", 16);

        TorchTabularMLP model(table.width, hidden_size, table.num_classes);
        train_torch_rows(table.config, rows, rows, table.split.train, table.split.test, model, false, torch_device(),
            table.results_dir, "tabular_mlp_torch_final.pt", table.class_names, table.num_classes, 32, 20, 0.05F,
            0.0005F);
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "Tabular Torch classification failed: %s\n", exception.what());
        return 1;
    }
}
