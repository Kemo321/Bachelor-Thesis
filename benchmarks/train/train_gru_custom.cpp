#include "common/experiment_config.hpp"
#include "common/indexed_train_custom.hpp"

#include "DeepLearnLib/loaders/Ecg200Loader.hpp"
#include "models/GruClassifier.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace
{

auto labels_as_float(const std::vector<int>& labels) -> std::vector<float>
{
    std::vector<float> values(labels.size());
    for (std::size_t index = 0; index < labels.size(); ++index)
    {
        values[index] = static_cast<float>(labels[index]);
    }
    return values;
}

auto ecg_rows(const Ecg200Data& data, const std::vector<float>& labels) -> HostRows
{
    HostRows rows;
    rows.values = &data.values;
    rows.labels = &labels;
    rows.rows = data.size();
    rows.width = data.length;
    rows.tail = { data.length, 1 };
    return rows;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::string pipeline_name = pipeline_name_from_args(argc, argv, "ecg200");
        const NamedPipeline pipeline = open_pipeline(pipeline_name, "data/ecg200", "results/ecg200");
        const int num_classes = pipeline.config.value("num_classes", 2);
        const int hidden_size = pipeline.config.value("hidden_size", 32);

        const Ecg200Data train_data = load_ecg200_file(find_ecg200_split(pipeline.data_root, "TRAIN"));
        const Ecg200Data test_data = load_ecg200_file(find_ecg200_split(pipeline.data_root, "TEST"));
        if (train_data.length != test_data.length || train_data.length < 1
            || train_data.class_names != test_data.class_names)
        {
            throw std::runtime_error("ECG200 train and test splits do not match");
        }
        if (static_cast<int>(train_data.class_names.size()) != num_classes)
        {
            throw std::runtime_error("ECG200 class labels do not match the experiment");
        }

        const std::vector<float> train_labels = labels_as_float(train_data.labels);
        const std::vector<float> test_labels = labels_as_float(test_data.labels);
        std::vector<std::string> class_names = train_data.class_names;
        if (pipeline.config.contains("class_names")
            && static_cast<int>(pipeline.config.at("class_names").size()) == num_classes)
        {
            class_names = pipeline.config.at("class_names").get<std::vector<std::string>>();
        }

        GruClassifier model(1, hidden_size, num_classes);
        auto layers = model.get_all_layers();
        train_custom_rows(
            pipeline.config, ecg_rows(train_data, train_labels), ecg_rows(test_data, test_labels),
            identity_order(train_data.size()), identity_order(test_data.size()),
            [&](const dl::Tensor& input)
            { return model.forward_logits(input); }, layers, true, pipeline.results_dir,
            "gru_classifier_final.bin", class_names, num_classes, 16, 80, 0.001F, 0.0F);
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "ECG200 classification failed: %s\n", exception.what());
        return 1;
    }
}
