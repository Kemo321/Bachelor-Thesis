#include "test_helpers.hpp"

#include "DeepLearnLib/CSVLoader.hpp"
#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/Softmax.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cmath>
#include <chrono>
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

TEST_F(GenericComponentTest, MseLossIsZeroWhenPredictionMatchesTarget)
{
    // Given: Identical prediction and target tensors
    const std::vector<float> values = { 1.0F, -2.0F, 0.5F, 3.0F };
    Tensor target = Tensor::from_host({ 2, 2 }, values, Device::GPU);
    Tensor prediction = Tensor::from_host({ 2, 2 }, values, Device::GPU);

    // When: MSE is evaluated
    Tensor loss = MSELoss::loss(target, prediction);
    synchronize_device();

    // Then: The scalar loss is zero
    expect_near_vector(loss.to_host(), { 0.0F });
}

TEST_F(GenericComponentTest, MseLossDerivativeScalesByTwoOverN)
{
    // Given: A constant offset between prediction and target
    Tensor target = Tensor::from_host({ 2 }, { 0.0F, 0.0F }, Device::GPU);
    Tensor prediction = Tensor::from_host({ 2 }, { 1.0F, 1.0F }, Device::GPU);

    // When: The MSE gradient is computed
    Tensor gradient = MSELoss::loss_derivative(target, prediction);
    synchronize_device();

    // Then: dL/dpred = 2 * (pred - target) / N = 1
    expect_near_vector(gradient.to_host(), { 1.0F, 1.0F });
}

TEST_F(GenericComponentTest, CrossEntropyPrefersTheCorrectOneHotClass)
{
    // Given: Logits that peak on class 1 and a matching one-hot target
    Tensor target = Tensor::from_host({ 1, 3 }, { 0.0F, 1.0F, 0.0F }, Device::GPU);
    Tensor confident = Tensor::from_host({ 1, 3 }, { 0.0F, 8.0F, 0.0F }, Device::GPU);
    Tensor uniform = Tensor::from_host({ 1, 3 }, { 0.0F, 0.0F, 0.0F }, Device::GPU);

    // When: Cross-entropy is evaluated for both predictions
    const float confident_loss = CrossEntropyLoss::loss(target, confident).to_host()[0];
    const float uniform_loss = CrossEntropyLoss::loss(target, uniform).to_host()[0];

    // Then: The confident logit row has a strictly smaller loss
    EXPECT_LT(confident_loss, uniform_loss);
    EXPECT_GT(confident_loss, 0.0F);
}

TEST_F(GenericComponentTest, CrossEntropyGradientIsSoftmaxMinusTargetOverBatch)
{
    // Given: Zero logits and a one-hot target on a batch of 1
    Tensor target = Tensor::from_host({ 1, 2 }, { 1.0F, 0.0F }, Device::GPU);
    Tensor logits = Tensor::from_host({ 1, 2 }, { 0.0F, 0.0F }, Device::GPU);

    // When: The fused softmax-cross-entropy gradient is computed
    Tensor gradient = CrossEntropyLoss::loss_derivative(target, logits);
    synchronize_device();

    // Then: softmax = [0.5, 0.5], so dL = softmax - target
    expect_near_vector(gradient.to_host(), { -0.5F, 0.5F });
}

TEST_F(GenericComponentTest, SoftmaxRowsSumToOneAndBackwardIsFinite)
{
    // Given: A rank-2 logit batch
    Tensor logits = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 1.0F }, Device::GPU);
    Softmax softmax;

    // When: Softmax is applied and a unit gradient is backpropagated
    Tensor probabilities = softmax.forward(logits);
    synchronize_device();
    Tensor grad_output = Tensor::from_host({ 2, 3 }, std::vector<float>(6, 1.0F), Device::GPU);
    Tensor grad_input = softmax.backward(grad_output);
    synchronize_device();

    // Then: Each row sums to 1 and the input gradient is finite
    const std::vector<float> host = probabilities.to_host();
    EXPECT_NEAR(host[0] + host[1] + host[2], 1.0F, kEpsilon);
    EXPECT_NEAR(host[3] + host[4] + host[5], 1.0F, kEpsilon);
    expect_all_finite(grad_input.to_host());
}

TEST_F(GenericComponentTest, SoftmaxRejectsCpuInputAndBackwardWithoutForward)
{
    // Given: A CPU tensor and a Softmax layer that has not run forward
    Tensor cpu_input = Tensor::from_host({ 1, 2 }, { 0.0F, 1.0F }, Device::CPU);
    Softmax softmax;

    // When / Then: Forward on CPU throws
    EXPECT_THROW(static_cast<void>(softmax.forward(cpu_input)), std::runtime_error);

    // When / Then: Backward without forward throws
    Tensor grad = Tensor::from_host({ 1, 2 }, { 1.0F, 0.0F }, Device::GPU);
    EXPECT_THROW(static_cast<void>(softmax.backward(grad)), std::runtime_error);
}

TEST_F(GenericComponentTest, CsvLoaderReadsFeaturesAndTargets)
{
    // Given: A rectangular float CSV with a header and two target columns
    const auto csv_path = std::filesystem::temp_directory_path()
        / ("dllib_csv_loader_test_"
            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv");
    {
        std::ofstream stream(csv_path);
        stream << "f0,f1,t0,t1\n";
        stream << "1.0,2.0,0.5,1.5\n";
        stream << "3.0,4.0,2.5,3.5\n";
    }

    // When: The file is loaded onto the GPU
    CSVLoader loader(csv_path.string(), 2, true);
    synchronize_device();

    // Then: Features and targets match the file layout
    EXPECT_EQ(loader.size(), 2U);
    EXPECT_EQ(loader.features().get_shape(), (std::vector<int> { 2, 2 }));
    EXPECT_EQ(loader.targets().get_shape(), (std::vector<int> { 2, 2 }));
    expect_near_vector(loader.features().to_host(), { 1.0F, 2.0F, 3.0F, 4.0F });
    expect_near_vector(loader.targets().to_host(), { 0.5F, 1.5F, 2.5F, 3.5F });
    std::filesystem::remove(csv_path);
}

TEST(CsvLoaderCpuTest, MissingFileThrows)
{
    // Given: A path that does not exist
    const std::string missing = "definitely-missing-dllib-dataset.csv";

    // When / Then: Construction throws before any GPU allocation
    EXPECT_THROW(CSVLoader(missing, 1, false), std::runtime_error);
}
