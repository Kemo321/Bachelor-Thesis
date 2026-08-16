#include "test_helpers.hpp"

#include "DeepLearnLib/BatchNorm2d.hpp"
#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/FusedCBR2d.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

class FusedCBR2dTest : public GpuTest
{
};

TEST_F(FusedCBR2dTest, EvalMatchesUnfusedConvBnLeaky)
{
    // Given: Identical conv weights and identity BatchNorm running stats
    FusedCBR2d fused(1, 1, 2, 1, 0, 0.1F);
    Conv2d conv(1, 1, 2, 1, 0);
    BatchNorm2d bn(1);
    LeakyReLU leaky(0.1F);
    std::map<std::string, Tensor> conv_params;
    set_named_parameter(conv_params, "weights", { 1, 1, 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F });
    set_named_parameter(conv_params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(conv_params);
    std::map<std::string, Tensor> bn_params;
    set_named_parameter(bn_params, "gamma", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(bn_params, "beta", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(bn_params, "running_mean", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(bn_params, "running_var", { 1, 1, 1, 1 }, { 1.0F });
    bn.set_parameters(bn_params);
    std::map<std::string, Tensor> fused_set;
    set_named_parameter(fused_set, "weights", { 1, 1, 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F });
    set_named_parameter(fused_set, "bias", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(fused_set, "gamma", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(fused_set, "beta", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(fused_set, "running_mean", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(fused_set, "running_var", { 1, 1, 1, 1 }, { 1.0F });
    fused.set_parameters(fused_set);
    fused.eval();
    bn.eval();
    Tensor input = Tensor::from_host({ 1, 1, 3, 3 }, std::vector<float>(9, 1.0F), Device::GPU);

    // When: Fused and sequential CBR blocks run in eval mode
    Tensor fused_out = fused.forward(input);
    Tensor sequential = leaky.forward(bn.forward(conv.forward(input)));
    synchronize_device();

    // Then: Both paths match and each 2x2 window is leaky(4 / sqrt(1+eps))
    EXPECT_EQ(fused_out.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
    expect_near_vector(fused_out.to_host(), sequential.to_host(), kLooseEpsilon);
}

TEST_F(FusedCBR2dTest, TrainingForwardBackwardAndStepAreFinite)
{
    // Given: A training fused CBR block with a mixed-sign input
    FusedCBR2d fused(1, 2, 3, 1, 1, 0.1F);
    fused.train();
    fused.learning_rate = 0.01F;
    Tensor input = Tensor::from_host({ 2, 1, 4, 4 }, std::vector<float>(32, 0.5F), Device::GPU);

    // When: Forward, backward, and an optimizer step run
    Tensor output = fused.forward(input);
    synchronize_device();
    Tensor grad = Tensor::from_host(output.get_shape(), std::vector<float>(output.get_size(), 1.0F), Device::GPU);
    Tensor grad_input = fused.backward(grad);
    const float weight_before = fused.get_parameters().at("weights").to_host().front();
    fused.step();
    synchronize_device();

    // Then: Activations and gradients are finite and convolution weights move
    expect_all_finite(output.to_host());
    EXPECT_EQ(grad_input.get_shape(), input.get_shape());
    expect_all_finite(grad_input.to_host());
    const float weight_after = fused.get_parameters().at("weights").to_host().front();
    EXPECT_NE(weight_after, weight_before);
}

TEST_F(FusedCBR2dTest, AffineEvalScaleAndLeakySlope)
{
    // Given: Eval BatchNorm with gamma=2, beta=-1 and a 1x1 convolution of ones
    FusedCBR2d fused(1, 1, 1, 1, 0, 0.1F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(params, "gamma", { 1, 1, 1, 1 }, { 2.0F });
    set_named_parameter(params, "beta", { 1, 1, 1, 1 }, { -1.0F });
    set_named_parameter(params, "running_mean", { 1, 1, 1, 1 }, { 0.0F });
    set_named_parameter(params, "running_var", { 1, 1, 1, 1 }, { 1.0F });
    fused.set_parameters(params);
    fused.eval();
    Tensor input = Tensor::from_host({ 1, 1, 1, 2 }, { -1.0F, 1.0F }, Device::GPU);

    // When: The fused forward pass is computed
    Tensor output = fused.forward(input);
    synchronize_device();

    // Then: y = leaky(2 * x / sqrt(1+eps) - 1)
    const float scale = 2.0F / std::sqrt(1.0F + 1e-5F);
    const float left = (scale * -1.0F) - 1.0F;
    const float right = (scale * 1.0F) - 1.0F;
    expect_near_vector(output.to_host(), { left * 0.1F, right }, kLooseEpsilon);
}

TEST_F(FusedCBR2dTest, InvalidConstructorArgumentsThrow)
{
    // Given: Illegal convolution or BatchNorm hyperparameters
    // When: FusedCBR2d is constructed
    // Then: Construction throws
    EXPECT_THROW(FusedCBR2d(0, 1, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(FusedCBR2d(1, 0, 1, 1, 0), std::runtime_error);
    EXPECT_THROW(FusedCBR2d(1, 1, 0, 1, 0), std::runtime_error);
    EXPECT_THROW(FusedCBR2d(1, 1, 1, 1, 0, 0.1F, -1e-5F), std::runtime_error);
    EXPECT_THROW(FusedCBR2d(1, 1, 1, 1, 0, 0.1F, 1e-5F, 1.5F), std::runtime_error);
}

TEST_F(FusedCBR2dTest, BackwardWithoutTrainingForwardThrows)
{
    // Given: An eval fused block and a training block that has not run forward
    FusedCBR2d eval_block(1, 1, 1, 1, 0);
    eval_block.eval();
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor output = eval_block.forward(input);
    Tensor grad = Tensor::from_host(output.get_shape(), std::vector<float>(output.get_size(), 1.0F), Device::GPU);
    FusedCBR2d idle(1, 1, 1, 1, 0);

    // When: Backward is invoked
    // Then: Both cases throw
    EXPECT_THROW(eval_block.backward(grad), std::runtime_error);
    EXPECT_THROW(idle.backward(grad), std::runtime_error);
}

TEST_F(FusedCBR2dTest, ParameterRoundTrip)
{
    // Given: Custom convolution and BatchNorm parameters
    FusedCBR2d fused(1, 1, 1, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 1, 1 }, { 0.25F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { -0.5F });
    set_named_parameter(params, "gamma", { 1, 1, 1, 1 }, { 1.5F });
    set_named_parameter(params, "beta", { 1, 1, 1, 1 }, { 0.75F });
    set_named_parameter(params, "running_mean", { 1, 1, 1, 1 }, { 0.1F });
    set_named_parameter(params, "running_var", { 1, 1, 1, 1 }, { 2.0F });
    fused.set_parameters(params);

    // When: Parameters are read back
    auto restored = fused.get_parameters();
    fused.to(Device::GPU);

    // Then: Values match and CPU placement is rejected
    expect_near_vector(restored.at("weights").to_host(), { 0.25F });
    expect_near_vector(restored.at("bias").to_host(), { -0.5F });
    expect_near_vector(restored.at("gamma").to_host(), { 1.5F });
    expect_near_vector(restored.at("beta").to_host(), { 0.75F });
    expect_near_vector(restored.at("running_mean").to_host(), { 0.1F });
    expect_near_vector(restored.at("running_var").to_host(), { 2.0F });
    EXPECT_FLOAT_EQ(fused.leaky_slope(), 0.1F);
    EXPECT_THROW(fused.to(Device::CPU), std::runtime_error);
}
