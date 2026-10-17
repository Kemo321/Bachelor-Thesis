#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
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

