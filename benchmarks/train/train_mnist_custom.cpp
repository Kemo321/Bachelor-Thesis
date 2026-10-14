#include "common/experiment_config.hpp"
#include "common/image_train_custom.hpp"

#include "DeepLearnLib/loaders/PackedImageLoader.hpp"
#include "models/SimpleCNN.hpp"

#include <cstdio>
#include <stdexcept>

int main()
{
    try
    {
        const NamedPipeline pipeline = open_pipeline("mnist_classification", "data/mnist", "results/mnist");
        const int batch_size = pipeline.config.value("batch_size", 128);
        const std::filesystem::path train_bin = pipeline.data_root / "train.bin";
        const std::filesystem::path test_bin = pipeline.data_root / "test.bin";

        PackedImageLoader train_loader(train_bin.string(), batch_size, true);
        PackedImageLoader test_loader(test_bin.string(), batch_size, false);
        if (test_loader.num_classes() != train_loader.num_classes() || test_loader.channels() != train_loader.channels()
            || test_loader.height() != train_loader.height() || test_loader.width() != train_loader.width())
        {
            throw std::runtime_error("MNIST train/test packed files disagree on shape or class count");
        }
        if (train_loader.height() != train_loader.width())
        {
            throw std::runtime_error("SimpleCNN expects square MNIST images");
        }
        if (train_loader.size() != 60000 || test_loader.size() != 10000)
        {
            std::fprintf(stderr, "MNIST expected 60000 train / 10000 test images; got %zu / %zu\n", train_loader.size(),
                test_loader.size());
        }

        SimpleCNN model(train_loader.num_classes(), train_loader.height(), train_loader.channels());
        train_custom_classifier(pipeline.config, train_loader, test_loader, model, pipeline.results_dir,
            "simplecnn_mnist_final.bin", train_loader.class_names());
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "MNIST classification failed: %s\n", exception.what());
        return 1;
    }
}
