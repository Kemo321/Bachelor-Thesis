#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/layers/Softmax.hpp"
#include "DeepLearnLib/loaders/CSVLoader.hpp"
#include "DeepLearnLib/losses/Losses.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

class GenericComponentTest : public GpuTest
{
};

TEST_F(GenericComponentTest, CrossEntropyPrefersTheCorrectOneHotClass)
{

    Tensor target = Tensor::from_host({ 1, 3 }, { 0.0F, 1.0F, 0.0F }, Device::GPU);
    Tensor confident = Tensor::from_host({ 1, 3 }, { 0.0F, 8.0F, 0.0F }, Device::GPU);
    Tensor uniform = Tensor::from_host({ 1, 3 }, { 0.0F, 0.0F, 0.0F }, Device::GPU);

    const float confident_loss = CrossEntropyLoss::loss(target, confident).to_host()[0];
    const float uniform_loss = CrossEntropyLoss::loss(target, uniform).to_host()[0];

    EXPECT_LT(confident_loss, uniform_loss);
    EXPECT_GT(confident_loss, 0.0F);
}

TEST_F(GenericComponentTest, CrossEntropyGradientIsSoftmaxMinusTargetOverBatch)
{

    Tensor target = Tensor::from_host({ 1, 2 }, { 1.0F, 0.0F }, Device::GPU);
    Tensor logits = Tensor::from_host({ 1, 2 }, { 0.0F, 0.0F }, Device::GPU);

    Tensor gradient = CrossEntropyLoss::loss_derivative(target, logits);
    synchronize_device();

    expect_near_vector(gradient.to_host(), { -0.5F, 0.5F });
}

TEST_F(GenericComponentTest, SoftmaxRowsSumToOneAndBackwardIsFinite)
{

    Tensor logits = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 1.0F }, Device::GPU);
    Softmax softmax;

    Tensor probabilities = softmax.forward(logits);
    synchronize_device();
    Tensor grad_output = Tensor::from_host({ 2, 3 }, std::vector<float>(6, 1.0F), Device::GPU);
    Tensor grad_input = softmax.backward(grad_output);
    synchronize_device();

    const std::vector<float> host = probabilities.to_host();
    EXPECT_NEAR(host[0] + host[1] + host[2], 1.0F, kEpsilon);
    EXPECT_NEAR(host[3] + host[4] + host[5], 1.0F, kEpsilon);
    expect_all_finite(grad_input.to_host());
}

TEST_F(GenericComponentTest, SoftmaxRejectsCpuInputAndBackwardWithoutForward)
{

    Tensor cpu_input = Tensor::from_host({ 1, 2 }, { 0.0F, 1.0F }, Device::CPU);
    Softmax softmax;

    EXPECT_THROW(static_cast<void>(softmax.forward(cpu_input)), std::runtime_error);

    Tensor grad = Tensor::from_host({ 1, 2 }, { 1.0F, 0.0F }, Device::GPU);
    EXPECT_THROW(static_cast<void>(softmax.backward(grad)), std::runtime_error);
}

TEST_F(GenericComponentTest, CsvLoaderReadsFeaturesAndTargets)
{

    const auto csv_path = std::filesystem::temp_directory_path()
        / ("dllib_csv_loader_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv");
    {
        std::ofstream stream(csv_path);
        stream << "f0,f1,t0,t1\n";
        stream << "1.0,2.0,0.5,1.5\n";
        stream << "3.0,4.0,2.5,3.5\n";
    }

    CSVLoader loader(csv_path.string(), 2, true);
    synchronize_device();

    EXPECT_EQ(loader.size(), 2U);
    EXPECT_EQ(loader.features().get_shape(), (std::vector<int> { 2, 2 }));
    EXPECT_EQ(loader.targets().get_shape(), (std::vector<int> { 2, 2 }));
    expect_near_vector(loader.features().to_host(), { 1.0F, 2.0F, 3.0F, 4.0F });
    expect_near_vector(loader.targets().to_host(), { 0.5F, 1.5F, 2.5F, 3.5F });
    std::filesystem::remove(csv_path);
}

TEST(CsvLoaderCpuTest, MissingFileThrows)
{

    const std::string missing = "definitely-missing-dllib-dataset.csv";

    EXPECT_THROW(CSVLoader(missing, 1, false), std::runtime_error);
}
