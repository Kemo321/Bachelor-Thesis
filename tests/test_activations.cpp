#include "test_helpers.hpp"

#include "DeepLearnLib/Dropout.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cmath>
#include <vector>

using namespace dl;
using namespace dllib_test;

class ActivationTest : public GpuTest
{
};

TEST_F(ActivationTest, LeakyReLUForwardAndBackwardSlope)
{
    // Given: A LeakyReLU with negative slope 0.1 and a mixed-sign input
    constexpr float slope = 0.1F;
    LeakyReLU relu(slope);
    Tensor input = Tensor::from_host({ 5 }, { -2.0F, -1.0F, 0.0F, 1.0F, 2.0F }, Device::GPU);

    // When: The forward pass is computed
    Tensor output = relu.forward(input);
    synchronize_device();

    // Then: Negative values are scaled by the slope and positives are unchanged
    expect_near_vector(output.to_host(), { -0.2F, -0.1F, 0.0F, 1.0F, 2.0F });

    // When: Unit gradients are backpropagated
    Tensor grad_output = Tensor::from_host({ 5 }, std::vector<float>(5, 1.0F), Device::GPU);
    Tensor grad_input = relu.backward(grad_output);
    synchronize_device();

    // Then: The local gradient is slope for x <= 0 and 1 otherwise
    expect_near_vector(grad_input.to_host(), { slope, slope, slope, 1.0F, 1.0F });
}

TEST_F(ActivationTest, DropoutEvalPreservesValues)
{
    // Given: Dropout in eval mode with a 2x2 input
    Dropout dropout(0.5F);
    dropout.eval();
    const std::vector<float> host_input = { 1.0F, 2.0F, 3.0F, 4.0F };
    Tensor input = Tensor::from_host({ 2, 2 }, host_input, Device::GPU);

    // When: The forward pass is computed
    Tensor output = dropout.forward(input);
    synchronize_device();

    // Then: Values are unchanged
    expect_near_vector(output.to_host(), host_input);

    // When: Unit gradients are backpropagated
    Tensor grad_output = Tensor::from_host({ 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    Tensor grad_input = dropout.backward(grad_output);
    synchronize_device();

    // Then: Gradients pass through unchanged
    expect_near_vector(grad_input.to_host(), std::vector<float>(4, 1.0F));
}

TEST_F(ActivationTest, DropoutTrainingZerosAndScales)
{
    // Given: Dropout in train mode with keep probability 0.5 and a constant ones vector
    Dropout dropout(0.5F);
    dropout.train();
    Tensor input = Tensor::from_host({ 256 }, std::vector<float>(256, 1.0F), Device::GPU);

    // When: The forward pass is computed
    Tensor output = dropout.forward(input);
    synchronize_device();

    // Then: Dropped values are zero and kept values are inverted-dropout scaled to 2
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

    // When: Unit gradients are backpropagated
    Tensor grad_output = Tensor::from_host({ 256 }, std::vector<float>(256, 1.0F), Device::GPU);
    Tensor grad_input = dropout.backward(grad_output);
    synchronize_device();

    // Then: The input gradient is finite
    expect_all_finite(grad_input.to_host());
}

TEST_F(ActivationTest, LeakyReLUDefaultSlopeOnNegatives)
{
    // Given: The default LeakyReLU slope and an all-negative vector
    LeakyReLU relu;
    Tensor input = Tensor::from_host({ 3 }, { -10.0F, -1.0F, -0.5F }, Device::GPU);

    // When: The forward pass is computed
    Tensor output = relu.forward(input);
    synchronize_device();

    // Then: Values are scaled by 0.1
    expect_near_vector(output.to_host(), { -1.0F, -0.1F, -0.05F });
}

TEST_F(ActivationTest, LeakyReLUZeroSlopeIsRelu)
{
    // Given: A zero-slope LeakyReLU
    LeakyReLU relu(0.0F);
    Tensor input = Tensor::from_host({ 4 }, { -2.0F, -0.1F, 0.0F, 3.0F }, Device::GPU);

    // When: Forward and backward are computed
    Tensor output = relu.forward(input);
    Tensor grad = relu.backward(Tensor::from_host({ 4 }, std::vector<float>(4, 1.0F), Device::GPU));
    synchronize_device();

    // Then: Negatives become 0 and their local gradient is 0
    expect_near_vector(output.to_host(), { 0.0F, 0.0F, 0.0F, 3.0F });
    expect_near_vector(grad.to_host(), { 0.0F, 0.0F, 0.0F, 1.0F });
}

TEST_F(ActivationTest, LeakyReLURejectsCpuAndMismatchedBackward)
{
    // Given: A LeakyReLU, a CPU tensor, and a size-mismatched gradient
    LeakyReLU relu(0.1F);
    Tensor cpu = Tensor::from_host({ 2 }, { 1.0F, -1.0F }, Device::CPU);
    Tensor input = Tensor::from_host({ 2 }, { 1.0F, -1.0F }, Device::GPU);
    (void)relu.forward(input);
    Tensor wrong_grad = Tensor::from_host({ 3 }, { 1.0F, 1.0F, 1.0F }, Device::GPU);
    LeakyReLU idle(0.1F);

    // When: Illegal calls are made
    // Then: Each call throws
    EXPECT_THROW(relu.forward(cpu), std::runtime_error);
    EXPECT_THROW(relu.backward(wrong_grad), std::runtime_error);
    EXPECT_THROW(idle.backward(input), std::runtime_error);
}

TEST_F(ActivationTest, DropoutZeroProbabilityKeepsValues)
{
    // Given: Dropout with drop probability 0 in train mode
    Dropout dropout(0.0F);
    dropout.train();
    const std::vector<float> host_input = { 1.0F, -2.0F, 3.0F };
    Tensor input = Tensor::from_host({ 3 }, host_input, Device::GPU);

    // When: The forward pass is computed
    Tensor output = dropout.forward(input);
    synchronize_device();

    // Then: Every value is kept at scale 1
    expect_near_vector(output.to_host(), host_input);
}

TEST_F(ActivationTest, DropoutInvalidProbabilityThrows)
{
    // Given: Probabilities outside [0, 1)
    // When: Dropout is constructed
    // Then: Construction throws
    EXPECT_THROW(Dropout(1.0F), std::runtime_error);
    EXPECT_THROW(Dropout(1.5F), std::runtime_error);
    EXPECT_THROW(Dropout(-0.1F), std::runtime_error);
}

TEST_F(ActivationTest, DropoutRejectsCpuAndMismatchedBackward)
{
    // Given: Train-mode dropout and illegal tensors
    Dropout dropout(0.5F);
    dropout.train();
    Tensor cpu = Tensor::from_host({ 4 }, std::vector<float>(4, 1.0F), Device::CPU);
    Tensor input = Tensor::from_host({ 4 }, std::vector<float>(4, 1.0F), Device::GPU);
    (void)dropout.forward(input);
    Tensor wrong_grad = Tensor::from_host({ 2 }, { 1.0F, 1.0F }, Device::GPU);

    // When: Illegal calls are made
    // Then: Each call throws
    EXPECT_THROW(dropout.forward(cpu), std::runtime_error);
    EXPECT_THROW(dropout.backward(wrong_grad), std::runtime_error);
}

TEST_F(ActivationTest, DropoutTrainThenEvalDisablesMask)
{
    // Given: Dropout that already sampled a train mask
    Dropout dropout(0.5F);
    dropout.train();
    Tensor input = Tensor::from_host({ 32 }, std::vector<float>(32, 1.0F), Device::GPU);
    (void)dropout.forward(input);
    dropout.eval();

    // When: Eval forward is computed
    Tensor output = dropout.forward(input);
    synchronize_device();

    // Then: Values pass through unchanged
    expect_near_vector(output.to_host(), std::vector<float>(32, 1.0F));
}
