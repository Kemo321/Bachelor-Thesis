#include "test_helpers.hpp"

#include "DeepLearnLib/BatchNorm2d.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

class BatchNorm2dTest : public GpuTest
{
};

TEST_F(BatchNorm2dTest, TrainingNormalizesAndUpdatesRunningStats)
{
    // Given: A training BatchNorm2d layer and a four-element single-channel batch
    BatchNorm2d bn(1, 1e-5F, 0.1F);
    bn.train();
    Tensor input = Tensor::from_host({ 2, 1, 1, 2 }, { 1.0F, 3.0F, 5.0F, 7.0F }, Device::GPU);

    // When: The forward pass is computed
    Tensor output = bn.forward(input);
    synchronize_device();

    // Then: The output is finite, zero-mean, and running stats have moved
    EXPECT_EQ(output.get_shape(), input.get_shape());
    const std::vector<float> out_host = output.to_host();
    expect_all_finite(out_host);

    float mean = 0.0F;
    for (float value : out_host)
    {
        mean += value;
    }
    mean /= static_cast<float>(out_host.size());
    EXPECT_NEAR(mean, 0.0F, kLooseEpsilon);

    auto stats = bn.get_parameters();
    const std::vector<float> running_mean = stats.at("running_mean").to_host();
    ASSERT_EQ(running_mean.size(), 1U);
    EXPECT_GT(std::fabs(running_mean[0]), 0.0F);

    // When: Unit gradients are backpropagated
    Tensor grad_output = Tensor::from_host({ 2, 1, 1, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    Tensor grad_input = bn.backward(grad_output);
    synchronize_device();

    // Then: The input gradient matches the input shape and is finite
    EXPECT_EQ(grad_input.get_shape(), input.get_shape());
    expect_all_finite(grad_input.to_host());
}

TEST_F(BatchNorm2dTest, EvalUsesRunningStatistics)
{
    // Given: An eval BatchNorm2d with identity affine params and unit running variance
    BatchNorm2d bn(1);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "gamma", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(params, "beta", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(params, "running_mean", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(params, "running_var", { 1, 1, 1, 1 }, { 1.0F });
    bn.set_parameters(params);
    bn.eval();
    const std::vector<float> host_input = { 0.0F, 1.0F, -1.0F, 2.0F };
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, host_input, Device::GPU);

    // When: The forward pass is computed
    Tensor output = bn.forward(input);
    synchronize_device();

    // Then: Values are scaled by 1 / sqrt(running_var + eps)
    const float scale = 1.0F / std::sqrt(1.0F + 1e-5F);
    std::vector<float> expected;
    expected.reserve(host_input.size());
    for (float value : host_input)
    {
        expected.push_back(value * scale);
    }
    expect_near_vector(output.to_host(), expected, kLooseEpsilon);
}

TEST_F(BatchNorm2dTest, InvalidConstructorArgumentsThrow)
{
    // Given: Illegal channel count or epsilon
    // When: BatchNorm2d is constructed
    // Then: Construction throws
    EXPECT_THROW(BatchNorm2d(0), std::runtime_error);
    EXPECT_THROW(BatchNorm2d(-1), std::runtime_error);
    EXPECT_THROW(BatchNorm2d(1, -1e-5F), std::runtime_error);
}

TEST_F(BatchNorm2dTest, ForwardRejectsChannelMismatchAndCpuInput)
{
    // Given: A 2-channel BatchNorm and illegal inputs
    BatchNorm2d bn(2);
    Tensor wrong_channels = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    Tensor cpu = Tensor::from_host({ 1, 2, 2, 2 }, std::vector<float>(8, 1.0F), Device::CPU);
    Tensor rank3 = Tensor::from_host({ 2, 2, 2 }, std::vector<float>(8, 1.0F), Device::GPU);

    // When: Forward is invoked
    // Then: Each illegal input throws
    EXPECT_THROW(bn.forward(wrong_channels), std::runtime_error);
    EXPECT_THROW(bn.forward(cpu), std::runtime_error);
    EXPECT_THROW(bn.forward(rank3), std::runtime_error);
}

TEST_F(BatchNorm2dTest, BackwardRequiresTrainingForward)
{
    // Given: An eval-mode layer and a training layer that has not run forward
    BatchNorm2d eval_bn(1);
    eval_bn.eval();
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    (void)eval_bn.forward(input);
    Tensor grad = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    BatchNorm2d idle(1);

    // When: Backward is invoked
    // Then: Both cases throw
    EXPECT_THROW(eval_bn.backward(grad), std::runtime_error);
    EXPECT_THROW(idle.backward(grad), std::runtime_error);
}

TEST_F(BatchNorm2dTest, EvalAffineScaleAndShift)
{
    // Given: Eval BatchNorm with gamma=2 and beta=1 over unit running stats
    BatchNorm2d bn(1);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "gamma", { 1, 1, 1, 1 }, { 2.0F });
    set_named_parameter(params, "beta", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(params, "running_mean", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(params, "running_var", { 1, 1, 1, 1 }, { 1.0F });
    bn.set_parameters(params);
    bn.eval();
    Tensor input = Tensor::from_host({ 1, 1, 1, 2 }, { 0.0F, 1.0F }, Device::GPU);

    // When: The forward pass is computed
    Tensor output = bn.forward(input);
    synchronize_device();

    // Then: y = 2 * x / sqrt(1+eps) + 1
    const float scale = 2.0F / std::sqrt(1.0F + 1e-5F);
    expect_near_vector(output.to_host(), { 1.0F, scale + 1.0F }, kLooseEpsilon);
}

TEST_F(BatchNorm2dTest, ParameterRoundTripAndGpuTo)
{
    // Given: Custom affine parameters
    BatchNorm2d bn(1);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "gamma", { 1, 1, 1, 1 }, { 1.5F });
    set_named_parameter(params, "beta", { 1, 1, 1, 1 }, { -0.5F });
    set_named_parameter(params, "running_mean", { 1, 1, 1, 1 }, { 0.25F });
    set_named_parameter(params, "running_var", { 1, 1, 1, 1 }, { 2.0F });
    bn.set_parameters(params);

    // When: Parameters are read back
    auto restored = bn.get_parameters();
    bn.to(Device::GPU);

    // Then: Values match and CPU placement is rejected
    expect_near_vector(restored.at("gamma").to_host(), { 1.5F });
    expect_near_vector(restored.at("beta").to_host(), { -0.5F });
    expect_near_vector(restored.at("running_mean").to_host(), { 0.25F });
    expect_near_vector(restored.at("running_var").to_host(), { 2.0F });
    EXPECT_THROW(bn.to(Device::CPU), std::runtime_error);
}

TEST_F(BatchNorm2dTest, StepChangesAffineParameters)
{
    // Given: A training BatchNorm with a large learning rate
    BatchNorm2d bn(1);
    bn.learning_rate = 1.0F;
    bn.train();
    Tensor input = Tensor::from_host({ 2, 1, 1, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    (void)bn.forward(input);
    Tensor grad = Tensor::from_host({ 2, 1, 1, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    (void)bn.backward(grad);
    const float gamma_before = bn.get_parameters().at("gamma").to_host().front();

    // When: An optimizer step is applied
    bn.step();
    synchronize_device();

    // Then: Gamma is no longer the initialized 1
    const float gamma_after = bn.get_parameters().at("gamma").to_host().front();
    EXPECT_NE(gamma_after, gamma_before);
}
