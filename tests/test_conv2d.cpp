#include "test_helpers.hpp"

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

class Conv2dTest : public GpuTest
{
};

TEST_F(Conv2dTest, ForwardOutputShape)
{
    // Given: A padded 3x3 convolution mapping 3 input channels to 8 output channels
    Conv2d conv(3, 8, 3, 1, 1);
    Tensor input = Tensor::from_host({ 2, 3, 16, 16 }, std::vector<float>(2 * 3 * 16 * 16, 0.5F), Device::GPU);

    // When: The forward pass is computed
    Tensor output = conv.forward(input);
    synchronize_device();

    // Then: The output is NCHW 2x8x16x16 on GPU with finite values
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 2, 8, 16, 16 }));
    EXPECT_EQ(output.get_device(), Device::GPU);
    expect_all_finite(output.to_host());
}

TEST_F(Conv2dTest, OnesKernelMatchesWindowSum)
{
    // Given: A 2x2 ones kernel with zero bias and a 3x3 ones input
    Conv2d conv(1, 1, 2, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 1, 3, 3 }, std::vector<float>(9, 1.0F), Device::GPU);

    // When: The forward pass is computed
    Tensor output = conv.forward(input);
    synchronize_device();

    // Then: Each 2x2 window sums to 4
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
    expect_near_vector(output.to_host(), { 4.0F, 4.0F, 4.0F, 4.0F });
}

TEST_F(Conv2dTest, IdentityOneByOneAndBackward)
{
    // Given: A 1x1 identity convolution and a 2x2 input
    Conv2d conv(1, 1, 1, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(params);
    const std::vector<float> host_input = { 1.0F, 2.0F, 3.0F, 4.0F };
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, host_input, Device::GPU);

    // When: The forward pass is computed
    Tensor output = conv.forward(input);
    synchronize_device();

    // Then: The output matches the input
    expect_near_vector(output.to_host(), host_input);

    // When: Unit gradients are backpropagated
    Tensor grad_output = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    Tensor grad_input = conv.backward(grad_output);
    synchronize_device();

    // Then: The input gradient is ones with the original spatial shape
    EXPECT_EQ(grad_input.get_shape(), input.get_shape());
    expect_near_vector(grad_input.to_host(), std::vector<float>(4, 1.0F));
}

TEST_F(Conv2dTest, BackwardWithoutForwardThrows)
{
    // Given: A convolution that has not run forward yet
    Conv2d conv(1, 1, 1, 1, 0);
    Tensor grad_output = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);

    // When: Backward is invoked without a cached forward
    // Then: The layer throws
    EXPECT_THROW(conv.backward(grad_output), std::runtime_error);
}

TEST_F(Conv2dTest, InvalidConstructorArgumentsThrow)
{
    // Given: Non-positive convolution hyperparameters
    // When: Conv2d is constructed
    // Then: Construction throws
    EXPECT_THROW(Conv2d(0, 1, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(Conv2d(1, 0, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(Conv2d(1, 1, 0, 1, 0), std::runtime_error);
    EXPECT_THROW(Conv2d(1, 1, 1, 0, 0), std::runtime_error);
    EXPECT_THROW(Conv2d(1, 1, 1, 1, -1), std::runtime_error);
}

TEST_F(Conv2dTest, ForwardRejectsChannelMismatchAndCpuInput)
{
    // Given: A 2-channel convolution and illegal inputs
    Conv2d conv(2, 2, 1, 1, 0);
    Tensor wrong_channels = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    Tensor cpu = Tensor::from_host({ 1, 2, 2, 2 }, std::vector<float>(8, 1.0F), Device::CPU);
    Tensor rank3 = Tensor::from_host({ 2, 2, 2 }, std::vector<float>(8, 1.0F), Device::GPU);

    // When: Forward is invoked
    // Then: Each illegal input throws
    EXPECT_THROW(conv.forward(wrong_channels), std::runtime_error);
    EXPECT_THROW(conv.forward(cpu), std::runtime_error);
    EXPECT_THROW(conv.forward(rank3), std::runtime_error);
}

TEST_F(Conv2dTest, StrideTwoHalvesSpatialSize)
{
    // Given: A 1x1 convolution with stride 2
    Conv2d conv(1, 1, 1, 2, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 1, 4, 4 }, std::vector<float>(16, 1.0F), Device::GPU);

    // When: The forward pass is computed
    Tensor output = conv.forward(input);
    synchronize_device();

    // Then: Spatial size is 2x2
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
}

TEST_F(Conv2dTest, ZeroWeightsAddBias)
{
    // Given: Zero weights and a constant bias
    Conv2d conv(1, 1, 1, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.25F });
    conv.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 9.0F, 8.0F, 7.0F, 6.0F }, Device::GPU);

    // When: The forward pass is computed
    Tensor output = conv.forward(input);
    synchronize_device();

    // Then: Every output equals the bias
    expect_near_vector(output.to_host(), { 0.25F, 0.25F, 0.25F, 0.25F });
}

TEST_F(Conv2dTest, ParameterRoundTripAndGpuTo)
{
    // Given: Known weights and bias
    Conv2d conv(1, 1, 1, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 0.5F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { -0.25F });
    conv.set_parameters(params);

    // When: Parameters are read back and to(GPU) is called
    auto restored = conv.get_parameters();
    conv.to(Device::GPU);

    // Then: Values match and CPU placement is rejected
    expect_near_vector(restored.at("weights").to_host(), { 0.5F });
    expect_near_vector(restored.at("bias").to_host(), { -0.25F });
    EXPECT_THROW(conv.to(Device::CPU), std::runtime_error);
}

TEST_F(Conv2dTest, StepUpdatesWeightsAfterBackward)
{
    // Given: An identity 1x1 convolution with a large learning rate
    Conv2d conv(1, 1, 1, 1, 0);
    conv.learning_rate = 1.0F;
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);
    Tensor output = conv.forward(input);
    synchronize_device();
    ASSERT_EQ(output.get_size(), 1U);

    // When: A unit gradient is backpropagated and step is applied
    Tensor grad_output = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);
    (void)conv.backward(grad_output);
    const std::vector<float> before = conv.get_parameters().at("weights").to_host();
    conv.step();
    synchronize_device();

    // Then: The weight value changes
    const std::vector<float> after = conv.get_parameters().at("weights").to_host();
    EXPECT_NE(after[0], before[0]);
}

TEST_F(Conv2dTest, BackwardRejectsMismatchedGradientShape)
{
    // Given: A convolution that has already run forward
    Conv2d conv(1, 1, 1, 1, 0);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    (void)conv.forward(input);
    Tensor wrong_grad = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);

    // When: Backward is given the wrong spatial shape
    // Then: The layer throws
    EXPECT_THROW(conv.backward(wrong_grad), std::runtime_error);
}
