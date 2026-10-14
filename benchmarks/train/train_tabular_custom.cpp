#include "common/indexed_train_custom.hpp"
#include "common/tabular_pipeline.hpp"

#include "models/TabularMLP.hpp"

#include <cstdio>

int main(int argc, char** argv)
{
    try
    {
        const std::string pipeline_name = pipeline_name_from_args(argc, argv, "tabular_demo");
        const TabularRun table = load_tabular_run(pipeline_name);
        const HostRows rows = tabular_host_rows(table);
        const int hidden_size = table.config.value("hidden_size", 16);

        TabularMLP model(table.width, hidden_size, table.num_classes);
        auto layers = model.get_all_layers();
        train_custom_rows(
            table.config, rows, rows, table.split.train, table.split.test,
            [&](const dl::Tensor& input)
            { return model.forward_logits(input); }, layers, false, table.results_dir,
            "tabular_mlp_final.bin", table.class_names, table.num_classes, 32, 20, 0.05F, 0.0005F);
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "Tabular classification failed: %s\n", exception.what());
        return 1;
    }
}
