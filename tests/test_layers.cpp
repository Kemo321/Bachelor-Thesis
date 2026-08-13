#include "DeepLearnLib/BatchNorm2d.hpp"
#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/Dropout.hpp"
#include "DeepLearnLib/Flatten.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/LeakyReLU.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dl;

namespace
{
constexpr float kEpsilon = 1e-4F;
constexpr float kLooseEpsilon = 1e-3F;

auto has_cuda_device() -> bool
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        return false;
    }
    return count > 0;
}

auto synchronize_device() -> void
{
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

auto expect_near_vector(const std::vector<float>& actual, const std::vector<float>& expected, float epsilon = kEpsilon)
    -> void
{
    EXPECT_EQ(actual.size(), expected.size());
    if (actual.size() != expected.size())
    {
        return;
    }
    for (size_t index = 0; index < actual.size(); ++index)
    {
        EXPECT_NEAR(actual[index], expected[index], epsilon) << "mismatch at index " << index;
    }
}

auto expect_all_finite(const std::vector<float>& values) -> void
{
    for (size_t index = 0; index < values.size(); ++index)
    {
        EXPECT_TRUE(std::isfinite(values[index])) << "non-finite value at index " << index;
    }
}

auto set_named_parameter(std::map<std::string, Tensor>& params, const std::string& name, const std::vector<int>& shape,
                         const std::vector<float>& host) -> void
{
    params.emplace(name, Tensor::from_host(shape, host, Device::GPU));
}
} // namespace

class GpuLayerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!has_cuda_device())
        {
            GTEST_SKIP() << "No CUDA-capable device available";
        }
    }
};

TEST_F(GpuLayerTest, Conv2dForwardOutputShape)
{
    Conv2d conv(3, 8, 3, 1, 1);
    Tensor input = Tensor::from_host({ 2, 3, 16, 16 }, std::vector<float>(2 * 3 * 16 * 16, 0.5F), Device::GPU);

    Tensor output = conv.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int>{ 2, 8, 16, 16 }));
    EXPECT_EQ(output.get_device(), Device::GPU);
    expect_all_finite(output.to_host());
}

TEST_F(GpuLayerTest, Conv2dOnesKernelMatchesWindowSum)
{
    Conv2d conv(1, 1, 2, 1, 0);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 1, 1, 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F });
    set_named_parameter(params, "bias", { 1, 1, 1, 1 }, { 0.0F });
    conv.set_parameters(params);

    Tensor input = Tensor::from_host({ 1, 1, 3, 3 }, std::vector<float>(9, 1.0F), Device::GPU);
    Tensor output = conv.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int>{ 1, 1, 2, 2 }));
    expect_near_vector(output.to_host(), { 4.0F, 4.0F, 4.0F, 4.0F });
}

TEST_F(GpuLayerTest, Conv2dIdentityOneByOneAndBackward)
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

TEST_F(GpuLayerTest, Conv2dBackwardWithoutForwardThrows)
{
    Conv2d conv(1, 1, 1, 1, 0);
    Tensor grad_output = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    EXPECT_THROW(conv.backward(grad_output), std::runtime_error);
}

TEST_F(GpuLayerTest, MaxPool2dDownsamplesAndRoutesGradientToArgmax)
{
    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 3.0F, 2.0F, 0.0F }, Device::GPU);

    Tensor output = pool.forward(input);
    synchronize_device();
    EXPECT_EQ(output.get_shape(), (std::vector<int>{ 1, 1, 1, 1 }));
    expect_near_vector(output.to_host(), { 3.0F });

    Tensor grad_output = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);
    Tensor grad_input = pool.backward(grad_output);
    synchronize_device();

    EXPECT_EQ(grad_input.get_shape(), (std::vector<int>{ 1, 1, 2, 2 }));
    expect_near_vector(grad_input.to_host(), { 0.0F, 1.0F, 0.0F, 0.0F });
}

TEST_F(GpuLayerTest, MaxPool2dFourByFourWindows)
{
    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 1, 4, 4 },
                                     { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 8.0F, 7.0F, 6.0F, 5.0F,
                                       4.0F, 3.0F, 2.0F },
                                     Device::GPU);

    Tensor output = pool.forward(input);
    synchronize_device();
    EXPECT_EQ(output.get_shape(), (std::vector<int>{ 1, 1, 2, 2 }));
    expect_near_vector(output.to_host(), { 6.0F, 8.0F, 9.0F, 7.0F });

    Tensor grad_output = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F }, Device::GPU);
    Tensor grad_input = pool.backward(grad_output);
    synchronize_device();

    const std::vector<float> grad_host = grad_input.to_host();
    EXPECT_EQ(grad_host.size(), 16U);
    float grad_sum = 0.0F;
    for (float value : grad_host)
    {
        grad_sum += value;
        EXPECT_TRUE(value == 0.0F || std::fabs(value - 1.0F) < kEpsilon);
    }
    EXPECT_NEAR(grad_sum, 4.0F, kEpsilon);
}

TEST_F(GpuLayerTest, BatchNorm2dTrainingNormalizesAndUpdatesRunningStats)
{
    BatchNorm2d bn(1, 1e-5F, 0.1F);
    bn.train();

    Tensor input = Tensor::from_host({ 2, 1, 1, 2 }, { 1.0F, 3.0F, 5.0F, 7.0F }, Device::GPU);
    Tensor output = bn.forward(input);
    synchronize_device();

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

    Tensor grad_output = Tensor::from_host({ 2, 1, 1, 2 }, std::vector<float>(4, 1.0F), Device::GPU);
    Tensor grad_input = bn.backward(grad_output);
    synchronize_device();
    EXPECT_EQ(grad_input.get_shape(), input.get_shape());
    expect_all_finite(grad_input.to_host());
}

TEST_F(GpuLayerTest, BatchNorm2dEvalUsesRunningStatistics)
{
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
    Tensor output = bn.forward(input);
    synchronize_device();

    const float scale = 1.0F / std::sqrt(1.0F + 1e-5F);
    std::vector<float> expected;
    expected.reserve(host_input.size());
    for (float value : host_input)
    {
        expected.push_back(value * scale);
    }
    expect_near_vector(output.to_host(), expected, kLooseEpsilon);
}

TEST_F(GpuLayerTest, LeakyReLUForwardAndBackwardSlope)
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

TEST_F(GpuLayerTest, DropoutEvalPreservesValues)
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

TEST_F(GpuLayerTest, DropoutTrainingZerosAndScales)
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

TEST_F(GpuLayerTest, FlattenReshapesAndUnflattensGradient)
{
    Flatten flatten;
    const std::vector<int> input_shape = { 2, 3, 2, 2 };
    std::vector<float> host_input(24);
    for (int index = 0; index < 24; ++index)
    {
        host_input[static_cast<size_t>(index)] = static_cast<float>(index);
    }
    Tensor input = Tensor::from_host(input_shape, host_input, Device::GPU);

    Tensor output = flatten.forward(input);
    synchronize_device();
    EXPECT_EQ(output.get_shape(), (std::vector<int>{ 2, 12 }));
    expect_near_vector(output.to_host(), host_input);

    Tensor grad_output = Tensor::from_host({ 2, 12 }, host_input, Device::GPU);
    Tensor grad_input = flatten.backward(grad_output);
    synchronize_device();
    EXPECT_EQ(grad_input.get_shape(), input_shape);
    expect_near_vector(grad_input.to_host(), host_input);
}

TEST_F(GpuLayerTest, FullyConnectedForwardMatchesMatmulPlusBias)
{
    FullyConnected dense(2, 3, 0.0F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 2, 3 }, { 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F });
    set_named_parameter(params, "bias", { 1, 3 }, { 0.5F, -1.0F, 2.0F });
    dense.set_parameters(params);

    Tensor input = Tensor::from_host({ 1, 2 }, { 1.0F, 2.0F }, Device::GPU);
    Tensor output = dense.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int>{ 1, 3 }));
    expect_near_vector(output.to_host(), { 1.5F, 1.0F, 5.0F });
}

TEST_F(GpuLayerTest, FullyConnectedBackwardAndStepUpdateWeights)
{
    FullyConnected dense(2, 2, 0.0F);
    dense.learning_rate = 1.0F;
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 2, 2 }, { 0.0F, 0.0F, 0.0F, 0.0F });
    set_named_parameter(params, "bias", { 1, 2 }, { 0.0F, 0.0F });
    dense.set_parameters(params);

    Tensor input = Tensor::from_host({ 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F }, Device::GPU);
    Tensor output = dense.forward(input);
    synchronize_device();
    expect_near_vector(output.to_host(), { 0.0F, 0.0F, 0.0F, 0.0F });

    Tensor grad_output = Tensor::from_host({ 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F }, Device::GPU);
    Tensor grad_input = dense.backward(grad_output);
    synchronize_device();
    EXPECT_EQ(grad_input.get_shape(), (std::vector<int>{ 2, 2 }));
    expect_near_vector(grad_input.to_host(), { 0.0F, 0.0F, 0.0F, 0.0F });

    dense.step();
    auto updated = dense.get_parameters();
    synchronize_device();
    expect_near_vector(updated.at("weights").to_host(), { -1.0F, 0.0F, 0.0F, -1.0F }, kLooseEpsilon);
}
