#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/layers/Conv2d.hpp"

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

    Conv2d conv(3, 8, 3, 1, 1);
    Tensor input = Tensor::from_host({ 2, 3, 16, 16 }, std::vector<float>(2 * 3 * 16 * 16, 0.5F), Device::GPU);

    Tensor output = conv.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int> { 2, 8, 16, 16 }));
    EXPECT_EQ(output.get_device(), Device::GPU);
    expect_all_finite(output.to_host());
}

TEST_F(Conv2dTest, OnesKernelMatchesWindowSum)
{

    Conv2d conv(1, 1, 2, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 1, 3, 3 }, std::vector<float>(9, 1.0F), Device::GPU);

    Tensor output = conv.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
    expect_near_vector(output.to_host(), { 4.0F, 4.0F, 4.0F, 4.0F });
}

TEST_F(Conv2dTest, IdentityOneByOneAndBackward)
{

    Conv2d conv(1, 1, 1, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(params);
    const std::vector<float> host_input = { 1.0F, 2.0F, 3.0F, 4.0F };
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, host_input, Device::GPU);

    Tensor output = conv.forward(input);
    synchronize_device();

    expect_near_vector(output.to_host(), host_input);

    Tensor grad_output = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    Tensor grad_input = conv.backward(grad_output);
    synchronize_device();

    EXPECT_EQ(grad_input.get_shape(), input.get_shape());
    expect_near_vector(grad_input.to_host(), std::vector<float>(4, 1.0F));
}

TEST_F(Conv2dTest, BackwardWithoutForwardThrows)
{

    Conv2d conv(1, 1, 1, 1, 0);
    Tensor grad_output = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);

    EXPECT_THROW(conv.backward(grad_output), std::runtime_error);
}

TEST_F(Conv2dTest, InvalidConstructorArgumentsThrow)
{

    EXPECT_THROW(Conv2d(0, 1, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(Conv2d(1, 0, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(Conv2d(1, 1, 0, 1, 0), std::runtime_error);
    EXPECT_THROW(Conv2d(1, 1, 1, 0, 0), std::runtime_error);
    EXPECT_THROW(Conv2d(1, 1, 1, 1, -1), std::runtime_error);
}

TEST_F(Conv2dTest, ForwardRejectsChannelMismatchAndCpuInput)
{

    Conv2d conv(2, 2, 1, 1, 0);
    Tensor wrong_channels = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    Tensor cpu = Tensor::from_host({ 1, 2, 2, 2 }, std::vector<float>(8, 1.0F), Device::CPU);
    Tensor rank3 = Tensor::from_host({ 2, 2, 2 }, std::vector<float>(8, 1.0F), Device::GPU);

    EXPECT_THROW(conv.forward(wrong_channels), std::runtime_error);
    EXPECT_THROW(conv.forward(cpu), std::runtime_error);
    EXPECT_THROW(conv.forward(rank3), std::runtime_error);
}

TEST_F(Conv2dTest, StrideTwoHalvesSpatialSize)
{

    Conv2d conv(1, 1, 1, 2, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 1, 4, 4 }, std::vector<float>(16, 1.0F), Device::GPU);

    Tensor output = conv.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
}

TEST_F(Conv2dTest, ZeroWeightsAddBias)
{

    Conv2d conv(1, 1, 1, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.25F });
    conv.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 9.0F, 8.0F, 7.0F, 6.0F }, Device::GPU);

    Tensor output = conv.forward(input);
    synchronize_device();

    expect_near_vector(output.to_host(), { 0.25F, 0.25F, 0.25F, 0.25F });
}

TEST_F(Conv2dTest, ParameterRoundTripAndGpuTo)
{

    Conv2d conv(1, 1, 1, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 0.5F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { -0.25F });
    conv.set_parameters(params);

    auto restored = conv.get_parameters();
    conv.to(Device::GPU);

    expect_near_vector(restored.at("weights").to_host(), { 0.5F });
    expect_near_vector(restored.at("bias").to_host(), { -0.25F });
    EXPECT_THROW(conv.to(Device::CPU), std::runtime_error);
}

TEST_F(Conv2dTest, StepUpdatesWeightsAfterBackward)
{

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

    Tensor grad_output = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);
    (void)conv.backward(grad_output);
    const std::vector<float> before = conv.get_parameters().at("weights").to_host();
    conv.step();
    synchronize_device();

    const std::vector<float> after = conv.get_parameters().at("weights").to_host();
    EXPECT_NE(after[0], before[0]);
}

TEST_F(Conv2dTest, BackwardRejectsMismatchedGradientShape)
{

    Conv2d conv(1, 1, 1, 1, 0);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    (void)conv.forward(input);
    Tensor wrong_grad = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);

    EXPECT_THROW(conv.backward(wrong_grad), std::runtime_error);
}
