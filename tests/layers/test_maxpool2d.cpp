#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/layers/MaxPool2d.hpp"

#include <cmath>
#include <vector>

using namespace dl;
using namespace dllib_test;

class MaxPool2dTest : public GpuTest
{
};

TEST_F(MaxPool2dTest, DownsamplesAndRoutesGradientToArgmax)
{

    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 3.0F, 2.0F, 0.0F }, Device::GPU);

    Tensor output = pool.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 1, 1, 1 }));
    expect_near_vector(output.to_host(), { 3.0F });

    Tensor grad_output = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);
    Tensor grad_input = pool.backward(grad_output);
    synchronize_device();

    EXPECT_EQ(grad_input.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
    expect_near_vector(grad_input.to_host(), { 0.0F, 1.0F, 0.0F, 0.0F });
}

TEST_F(MaxPool2dTest, FourByFourWindows)
{

    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 1, 4, 4 },
        { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 8.0F, 7.0F, 6.0F, 5.0F,
            4.0F, 3.0F, 2.0F },
        Device::GPU);

    Tensor output = pool.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
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

TEST_F(MaxPool2dTest, InvalidConstructorArgumentsThrow)
{

    EXPECT_THROW(MaxPool2d(0, 2), std::runtime_error);
    EXPECT_THROW(MaxPool2d(2, 0), std::runtime_error);
    EXPECT_THROW(MaxPool2d(-1, 1), std::runtime_error);
}

TEST_F(MaxPool2dTest, ForwardRejectsCpuAndNonNchw)
{

    MaxPool2d pool(2, 2);
    Tensor cpu = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::CPU);
    Tensor rank3 = Tensor::from_host({ 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    EXPECT_THROW(pool.forward(cpu), std::runtime_error);
    EXPECT_THROW(pool.forward(rank3), std::runtime_error);
}

TEST_F(MaxPool2dTest, BackwardWithoutForwardThrows)
{

    MaxPool2d pool(2, 2);
    Tensor grad = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);

    EXPECT_THROW(pool.backward(grad), std::runtime_error);
}

TEST_F(MaxPool2dTest, BackwardRejectsMismatchedGradientShape)
{

    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    (void)pool.forward(input);
    Tensor wrong_grad = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);

    EXPECT_THROW(pool.backward(wrong_grad), std::runtime_error);
}

TEST_F(MaxPool2dTest, MultiChannelPoolsIndependently)
{

    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 2, 2, 2 }, { 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 5.0F }, Device::GPU);

    Tensor output = pool.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 2, 1, 1 }));
    expect_near_vector(output.to_host(), { 1.0F, 5.0F });
}

TEST_F(MaxPool2dTest, RepeatedForwardReusesDescriptors)
{

    MaxPool2d pool(2, 2);
    Tensor first = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor second = Tensor::from_host({ 1, 1, 2, 2 }, { 4.0F, 3.0F, 2.0F, 1.0F }, Device::GPU);

    Tensor out1 = pool.forward(first);
    Tensor out2 = pool.forward(second);
    synchronize_device();

    expect_near_vector(out1.to_host(), { 4.0F });
    expect_near_vector(out2.to_host(), { 4.0F });
}
