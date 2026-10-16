#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/layers/Dropout.hpp"
#include "DeepLearnLib/layers/LeakyReLU.hpp"

#include <cmath>
#include <vector>

using namespace dl;
using namespace dllib_test;

class ActivationTest : public GpuTest
{
};

TEST_F(ActivationTest, LeakyReLUForwardAndBackwardSlope)
{

    constexpr float slope = 0.1F;
    LeakyReLU relu(slope);
    Tensor input = Tensor::from_host({ 5 }, { -2.0F, -1.0F, 0.0F, 1.0F, 2.0F }, Device::GPU);

    Tensor output = relu.forward(input);
    synchronize_device();

    expect_near_vector(output.to_host(), { -0.2F, -0.1F, 0.0F, 1.0F, 2.0F });

    Tensor grad_output = Tensor::from_host({ 5 }, std::vector<float>(5, 1.0F), Device::GPU);
    Tensor grad_input = relu.backward(grad_output);
    synchronize_device();

    expect_near_vector(grad_input.to_host(), { slope, slope, slope, 1.0F, 1.0F });
}

TEST_F(ActivationTest, DropoutEvalPreservesValues)
{

    Dropout dropout(0.5F);
    dropout.eval();
    const std::vector<float> host_input = { 1.0F, 2.0F, 3.0F, 4.0F };
    Tensor input = Tensor::from_host({ 2, 2 }, host_input, Device::GPU);

    Tensor output = dropout.forward(input);
    synchronize_device();

    expect_near_vector(output.to_host(), host_input);

    Tensor grad_output = Tensor::from_host({ 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    Tensor grad_input = dropout.backward(grad_output);
    synchronize_device();

    expect_near_vector(grad_input.to_host(), std::vector<float>(4, 1.0F));
}

TEST_F(ActivationTest, DropoutTrainingZerosAndScales)
{

    Dropout dropout(0.5F);
    dropout.train();
    Tensor input = Tensor::from_host({ 256 }, std::vector<float>(256, 1.0F), Device::GPU);

    Tensor output = dropout.forward(input);
    synchronize_device();

    const std::vector<float> host = output.to_host();
    int zero_count = 0;
    int kept_count = 0;
    for (float value : host)
    {
        if (std::fabs(value) < kEpsilon)
        {
            ++zero_count;
        }
        else
        {
            EXPECT_NEAR(value, 2.0F, kEpsilon);
            ++kept_count;
        }
    }
    EXPECT_GT(zero_count, 0);
    EXPECT_GT(kept_count, 0);

    Tensor grad_output = Tensor::from_host({ 256 }, std::vector<float>(256, 1.0F), Device::GPU);
    Tensor grad_input = dropout.backward(grad_output);
    synchronize_device();

    expect_all_finite(grad_input.to_host());
}

TEST_F(ActivationTest, LeakyReLUDefaultSlopeOnNegatives)
{

    LeakyReLU relu;
    Tensor input = Tensor::from_host({ 3 }, { -10.0F, -1.0F, -0.5F }, Device::GPU);

    Tensor output = relu.forward(input);
    synchronize_device();

    expect_near_vector(output.to_host(), { -1.0F, -0.1F, -0.05F });
}

TEST_F(ActivationTest, LeakyReLUZeroSlopeIsRelu)
{

    LeakyReLU relu(0.0F);
    Tensor input = Tensor::from_host({ 4 }, { -2.0F, -0.1F, 0.0F, 3.0F }, Device::GPU);

    Tensor output = relu.forward(input);
    Tensor grad = relu.backward(Tensor::from_host({ 4 }, std::vector<float>(4, 1.0F), Device::GPU));
    synchronize_device();

    expect_near_vector(output.to_host(), { 0.0F, 0.0F, 0.0F, 3.0F });
    expect_near_vector(grad.to_host(), { 0.0F, 0.0F, 0.0F, 1.0F });
}

TEST_F(ActivationTest, LeakyReLURejectsCpuAndMismatchedBackward)
{

    LeakyReLU relu(0.1F);
    Tensor cpu = Tensor::from_host({ 2 }, { 1.0F, -1.0F }, Device::CPU);
    Tensor input = Tensor::from_host({ 2 }, { 1.0F, -1.0F }, Device::GPU);
    (void)relu.forward(input);
    Tensor wrong_grad = Tensor::from_host({ 3 }, { 1.0F, 1.0F, 1.0F }, Device::GPU);
    LeakyReLU idle(0.1F);

    EXPECT_THROW(relu.forward(cpu), std::runtime_error);
    EXPECT_THROW(relu.backward(wrong_grad), std::runtime_error);
    EXPECT_THROW(idle.backward(input), std::runtime_error);
}

TEST_F(ActivationTest, DropoutZeroProbabilityKeepsValues)
{

    Dropout dropout(0.0F);
    dropout.train();
    const std::vector<float> host_input = { 1.0F, -2.0F, 3.0F };
    Tensor input = Tensor::from_host({ 3 }, host_input, Device::GPU);

    Tensor output = dropout.forward(input);
    synchronize_device();

    expect_near_vector(output.to_host(), host_input);
}

TEST_F(ActivationTest, DropoutInvalidProbabilityThrows)
{

    EXPECT_THROW(Dropout(1.0F), std::runtime_error);
    EXPECT_THROW(Dropout(1.5F), std::runtime_error);
    EXPECT_THROW(Dropout(-0.1F), std::runtime_error);
}

TEST_F(ActivationTest, DropoutRejectsCpuAndMismatchedBackward)
{

    Dropout dropout(0.5F);
    dropout.train();
    Tensor cpu = Tensor::from_host({ 4 }, std::vector<float>(4, 1.0F), Device::CPU);
    Tensor input = Tensor::from_host({ 4 }, std::vector<float>(4, 1.0F), Device::GPU);
    (void)dropout.forward(input);
    Tensor wrong_grad = Tensor::from_host({ 2 }, { 1.0F, 1.0F }, Device::GPU);

    EXPECT_THROW(dropout.forward(cpu), std::runtime_error);
    EXPECT_THROW(dropout.backward(wrong_grad), std::runtime_error);
}

TEST_F(ActivationTest, DropoutTrainThenEvalDisablesMask)
{

    Dropout dropout(0.5F);
    dropout.train();
    Tensor input = Tensor::from_host({ 32 }, std::vector<float>(32, 1.0F), Device::GPU);
    (void)dropout.forward(input);
    dropout.eval();

    Tensor output = dropout.forward(input);
    synchronize_device();

    expect_near_vector(output.to_host(), std::vector<float>(32, 1.0F));
}
