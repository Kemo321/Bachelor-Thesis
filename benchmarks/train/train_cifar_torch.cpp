#include "common/experiment_config.hpp"
#include "common/image_train_torch.hpp"
#include "models/torch/TorchSimpleCNN.hpp"

#include "DeepLearnLib/loaders/ClassificationLoader.hpp"

#include <cstdio>

int main()
{
    try
    {
        const NamedPipeline pipeline = open_pipeline("cifar10_classification", "data/cifar10", "results/cifar10");
        const int batch_size = pipeline.config.value("batch_size", 128);
        const int image_size = pipeline.config.value("image_size", 32);
        const int in_channels = pipeline.config.value("in_channels", 3);
        const std::string train_split = pipeline.config.value("train_split", "train");
        const std::string test_split = pipeline.config.value("test_split", "test");
        const torch::Device device = torch_device();

        ClassificationLoader train_loader(pipeline.data_root.string(), train_split, batch_size, image_size, true);
        const std::vector<std::string> class_names = train_loader.class_names();
        ClassificationLoader test_loader(
            pipeline.data_root.string(), test_split, batch_size, image_size, false, class_names);
        if (train_loader.size() != 50000 || test_loader.size() != 10000)
        {
            std::fprintf(stderr, "CIFAR-10 expected 50000 train / 10000 test images; got %zu / %zu\n",
                train_loader.size(), test_loader.size());
        }

        TorchSimpleCNN model(train_loader.num_classes(), image_size, in_channels);
        train_torch_classifier(pipeline.config, train_loader, test_loader, model, device, pipeline.results_dir,
            "simplecnn_cifar10_torch_final.pt", class_names);
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "CIFAR-10 Torch classification failed: %s\n", exception.what());
        return 1;
    }
}
