#include "test_helpers.hpp"

#include "DeepLearnLib/layers/FullyConnected.hpp"
#include "DeepLearnLib/losses/Losses.hpp"
#include "DeepLearnLib/core/Network.hpp"
#include "DeepLearnLib/utils/SafeMath.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <vector>

using namespace dl;
using namespace dllib_test;

class GpuNumericsTest : public GpuTest
{
};

TEST_F(GpuNumericsTest, CrossEntropyStaysFiniteForOverflowingLogits)
{
    // Given: Logits large enough that exp(x) would overflow without max-subtraction
    Tensor target = Tensor::from_host({ 1, 3 }, { 0.0F, 1.0F, 0.0F }, Device::GPU);
    Tensor logits = Tensor::from_host({ 1, 3 }, { 80.0F, 1000.0F, -80.0F }, Device::GPU);

    // When: Softmax-cross-entropy and its gradient are computed
    const float loss = CrossEntropyLoss::loss(target, logits).to_host().front();
    Tensor gradient = CrossEntropyLoss::loss_derivative(target, logits);
    synchronize_device();

    // Then: The loss is a finite non-negative scalar and the gradient is finite
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GE(loss, 0.0F);
    expect_all_finite(gradient.to_host());
}

TEST_F(GpuNumericsTest, CrossEntropyClampsNearZeroProbabilities)
{
    // Given: A one-hot target on a class whose logit is far below the others
    Tensor target = Tensor::from_host({ 1, 2 }, { 1.0F, 0.0F }, Device::GPU);
    Tensor logits = Tensor::from_host({ 1, 2 }, { -1.0e6F, 1.0e6F }, Device::GPU);

    // When: Cross-entropy is evaluated
    const float loss = CrossEntropyLoss::loss(target, logits).to_host().front();
    synchronize_device();

    // Then: log(prob) is clamped, so the loss is finite and near -log(kSafeEps)
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(loss, -std::log(kSafeEps), 1.0e-3F); // clamped loss stays within 1e-3 of -log(kSafeEps)
}

TEST_F(GpuNumericsTest, NetworkStoresConfigurableGradientClip)
{
    // Given: A dense layer wrapped in a network with a custom clip bound
    auto dense = std::make_shared<FullyConnected>(2, 2);
    const float clip_bound = 2.5F;

    // When: The network is constructed and the bound is later overwritten
    Network network({ dense }, 0.01F, clip_bound);

    // Then: The configured clip is stored and the setter updates it
    EXPECT_FLOAT_EQ(network.gradient_clip(), clip_bound);
    network.set_gradient_clip(7.0F);
    EXPECT_FLOAT_EQ(network.gradient_clip(), 7.0F);
}

// 100 and −50 are clipped to +/- 3, while 0 and 3 pass through.
TEST_F(GpuNumericsTest, ClipLossGradientBoundsEveryElement)
{
    // Given: A loss gradient with values far outside the clip window
    auto dense = std::make_shared<FullyConnected>(2, 2);
    Network network({ dense }, 0.01F, 3.0F);
    Tensor huge = Tensor::from_host({ 1, 4 }, { 100.0F, -50.0F, 0.0F, 3.0F }, Device::GPU);

    // When: The network clips the tensor to the configured bound
    Tensor clipped = network.clip_loss_gradient(huge);
    synchronize_device();
    constexpr float bound = 3.0F;

    // Then: Every value lies in [-bound, bound]
    const std::vector<float> host = clipped.to_host();
    ASSERT_EQ(host.size(), 4U);
    for (float value : host)
    {
        EXPECT_LE(std::fabs(value), bound + 1.0e-5F);
    }
    EXPECT_NEAR(host[0], bound, 1.0e-5F);
    EXPECT_NEAR(host[1], -bound, 1.0e-5F);
    EXPECT_NEAR(host[2], 0.0F, 1.0e-5F);
    EXPECT_NEAR(host[3], 3.0F, 1.0e-5F);
}

TEST_F(GpuNumericsTest, ParameterGradientClipBoundsDenseUpdates)
{
    // Given: A 1x1 dense layer with zero weights and a huge incoming gradient
    auto dense = std::make_shared<FullyConnected>(1, 1, 0.0F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1 }, { 0.0F });
    set_named_parameter(params, "bias", { 1, 1 }, { 0.0F });
    dense->set_parameters(params);
    dense->learning_rate = 1.0F;
    Network network({ dense }, 1.0F, 1.0F);
    Tensor input = Tensor::from_host({ 1, 1 }, { 1.0F }, Device::GPU);
    Tensor grad_output = Tensor::from_host({ 1, 1 }, { 1000.0F }, Device::GPU);

    // When: Backward accumulates a large weight gradient that is clipped before step
    (void)dense->forward(input);
    (void)dense->backward(grad_output);
    network.clip_parameter_gradients();
    dense->step();
    synchronize_device();

    // Then: The weight update is bounded by the clip value (plus a tiny decay term)
    const std::vector<float> weights = dense->get_parameters().at("weights").to_host();
    ASSERT_EQ(weights.size(), 1U);
    EXPECT_NEAR(weights[0], -1.0F, 1.0e-3F); // 1e-3 covers the weight-decay term around the clipped step of -1
}

TEST_F(GpuNumericsTest, HasNonFiniteDetectsInfAndNan)
{
    // Given: GPU tensors that contain Inf and NaN
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor inf_tensor = Tensor::from_host({ 2 }, { 1.0F, inf }, Device::GPU);
    Tensor nan_tensor = Tensor::from_host({ 2 }, { 0.0F, nan }, Device::GPU);
    Tensor finite_tensor = Tensor::from_host({ 2 }, { 1.0F, -2.0F }, Device::GPU);

    // When: Each tensor is scanned for non-finite values
    synchronize_device();

    // Then: Inf/NaN tensors report true and a finite tensor reports false
    EXPECT_TRUE(inf_tensor.has_non_finite());
    EXPECT_TRUE(nan_tensor.has_non_finite());
    EXPECT_FALSE(finite_tensor.has_non_finite());
}

TEST_F(GpuNumericsTest, DefaultNetworkGradientClipIsDisabled)
{
    // Given: A network constructed without an explicit clip argument
    auto dense = std::make_shared<FullyConnected>(2, 2);

    // When: The two-argument constructor runs
    Network network({ dense }, 0.01F);

    // Then: Clipping is off so custom SGD matches an unclipped Torch step
    EXPECT_FLOAT_EQ(network.gradient_clip(), kDefaultGradientClip);
    EXPECT_FLOAT_EQ(network.gradient_clip(), 0.0F);
}
