#include "test_helpers.hpp"

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/Precision.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

#include <map>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

namespace
{
constexpr float kHalfEpsilon = 1.0e-2F;
constexpr float kStaticLossScale = 1024.0F;
} // namespace

class PrecisionTest : public GpuTest
{
};

TEST_F(PrecisionTest, DefaultComputeIsFloat32WithUnitLossScale)
{
    // Given: The process-wide precision policy has not been switched to FP16
    MixedPrecisionGuard guard(false, 1.0F);

    // When: The active compute dtype and loss scale are queried
    // Then: Training stays in FP32 with an unscaled backward pass
    EXPECT_FALSE(mixed_precision_enabled());
    EXPECT_EQ(compute_dtype(), Dtype::Float32);
    EXPECT_FLOAT_EQ(loss_scale(), 1.0F);
}

TEST_F(PrecisionTest, JsonFp32FallbackDisablesMixedPrecision)
{
    // Given: Mixed precision is currently enabled
    MixedPrecisionGuard guard(true, kStaticLossScale);
    ASSERT_TRUE(mixed_precision_enabled());

    // When: Pipeline JSON requests an explicit FP32 fallback
    configure_precision(true, "fp32", kStaticLossScale);

    // Then: Compute stays FP32 and the loss scale is identity
    EXPECT_FALSE(mixed_precision_enabled());
    EXPECT_EQ(compute_dtype(), Dtype::Float32);
    EXPECT_FLOAT_EQ(loss_scale(), 1.0F);
    EXPECT_STREQ(dtype_name(compute_dtype()), "fp32");
}

TEST_F(PrecisionTest, HalfTensorRoundTripPreservesHostValues)
{
    // Given: Host floats copied into an FP16 GPU tensor
    const std::vector<float> host = { 0.5F, -1.0F, 2.0F, 4.0F };
    Tensor tensor = Tensor::from_host({ 2, 2 }, host, Device::GPU, 0, Dtype::Float16);

    // When: The tensor is read back to host
    synchronize_device();

    // Then: Storage is half-precision and values survive the round trip
    EXPECT_EQ(tensor.get_dtype(), Dtype::Float16);
    EXPECT_EQ(tensor.nbytes(), host.size() * element_size(Dtype::Float16));
    expect_near_vector(tensor.to_host(), host, kHalfEpsilon);
}

TEST_F(PrecisionTest, Conv2dFp16ForwardMatchesWindowSum)
{
    // Given: Mixed precision is on so Conv2d weights allocate as FP16 Tensor Core inputs
    MixedPrecisionGuard guard(true, kStaticLossScale);
    Conv2d conv(1, 1, 2, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 1, 3, 3 }, std::vector<float>(9, 1.0F), Device::GPU, 0, Dtype::Float16);

    // When: The forward pass runs with CUDNN_DATA_HALF descriptors
    Tensor output = conv.forward(input);
    synchronize_device();

    // Then: Each 2x2 window still sums to 4 and the activation is FP16
    EXPECT_EQ(output.get_dtype(), Dtype::Float16);
    EXPECT_EQ(conv.get_parameters().at("weights").get_dtype(), Dtype::Float16);
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
    expect_near_vector(output.to_host(), { 4.0F, 4.0F, 4.0F, 4.0F }, kHalfEpsilon);
}

TEST_F(PrecisionTest, FullyConnectedFp16ForwardMatchesMatmulPlusBias)
{
    // Given: An FP16 dense layer with known weights
    MixedPrecisionGuard guard(true, kStaticLossScale);
    FullyConnected dense(2, 3, 0.0F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 2, 3 }, { 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F });
    set_named_parameter(params, "bias", { 1, 3 }, { 0.5F, -1.0F, 2.0F });
    dense.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 2 }, { 1.0F, 2.0F }, Device::GPU, 0, Dtype::Float16);

    // When: The Tensor Core GEMM forward pass is computed
    Tensor output = dense.forward(input);
    synchronize_device();

    // Then: The result matches input @ weights + bias in FP16 storage
    EXPECT_EQ(output.get_dtype(), Dtype::Float16);
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 3 }));
    expect_near_vector(output.to_host(), { 1.5F, 1.0F, 5.0F }, kHalfEpsilon);
}

TEST_F(PrecisionTest, CrossEntropyLossScaleMultipliesGradients)
{
    // Given: Static loss scaling of 1024 and a one-hot classification pair
    MixedPrecisionGuard guard(true, kStaticLossScale);
    Tensor target = Tensor::from_host({ 1, 2 }, { 1.0F, 0.0F }, Device::GPU);
    Tensor logits = Tensor::from_host({ 1, 2 }, { 0.0F, 0.0F }, Device::GPU);

    // When: The fused softmax-cross-entropy gradient is computed
    Tensor gradient = CrossEntropyLoss::loss_derivative(target, logits);
    synchronize_device();

    // Then: softmax-[target] is multiplied by the loss scale before backward
    expect_near_vector(gradient.to_host(), { -0.5F * kStaticLossScale, 0.5F * kStaticLossScale });
}

TEST_F(PrecisionTest, StepUnscalesGradientsByLossScale)
{
    // Given: An FP16 dense layer whose backward grads are already loss-scaled
    MixedPrecisionGuard guard(true, kStaticLossScale);
    FullyConnected dense(2, 2, 0.0F);
    dense.learning_rate = 1.0F;
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 2, 2 }, { 0.0F, 0.0F, 0.0F, 0.0F });
    set_named_parameter(params, "bias", { 1, 2 }, { 0.0F, 0.0F });
    dense.set_parameters(params);
    Tensor input = Tensor::from_host({ 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F }, Device::GPU, 0, Dtype::Float16);
    Tensor unused = dense.forward(input);
    (void)unused;
    Tensor grad_output = Tensor::from_host({ 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F }, Device::GPU, 0, Dtype::Float16);
    Tensor unused_grad = dense.backward(grad_output);
    (void)unused_grad;

    // When: An optimizer step unscales by dividing the learning rate by loss_scale
    dense.step();
    auto updated = dense.get_parameters();
    synchronize_device();

    // Then: The effective update matches the FP32 step (lr / 1024)
    const float step = 1.0F / kStaticLossScale;
    expect_near_vector(updated.at("weights").to_host(), { -step, 0.0F, 0.0F, -step }, kHalfEpsilon);
}

TEST_F(PrecisionTest, YoloLossDerivativeHonorsLossScale)
{
    // Given: Mixed precision with a static scale and a trivial empty YOLO grid
    MixedPrecisionGuard guard(true, kStaticLossScale);
    constexpr int kGrid = 7;
    constexpr int kNumClasses = 20;
    constexpr int kAttributes = 10 + kNumClasses;
    const std::vector<float> zeros(static_cast<size_t>(kGrid * kGrid * kAttributes), 0.0F);
    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros, Device::GPU, 0, Dtype::Float16);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros, Device::GPU, 0, Dtype::Float16);

    // When: The YOLO gradient is computed
    Tensor gradient = YOLOLoss::loss_derivative(target, prediction, kNumClasses);
    synchronize_device();

    // Then: The result stays finite in FP16 and the logged loss remains unscaled
    EXPECT_EQ(gradient.get_dtype(), Dtype::Float16);
    expect_all_finite(gradient.to_host());
    Tensor loss = YOLOLoss::loss(target, prediction, kNumClasses);
    synchronize_device();
    ASSERT_EQ(loss.to_host().size(), 1U);
    EXPECT_NEAR(loss.to_host()[0], 0.0F, 1e-3F);
}
