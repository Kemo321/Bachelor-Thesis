#include "test_helpers.hpp"

#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cmath>
#include <vector>

using namespace dl;
using namespace dllib_test;

class MaxPool2dTest : public GpuTest
{
};

TEST_F(MaxPool2dTest, DownsamplesAndRoutesGradientToArgmax)
{
    // Given: A 2x2 pool and a 2x2 input whose maximum is at (0, 1)
    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 3.0F, 2.0F, 0.0F }, Device::GPU);

    // When: The forward pass is computed
    Tensor output = pool.forward(input);
    synchronize_device();

    // Then: The pooled value is 3 with shape 1x1x1x1
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 1, 1, 1 }));
    expect_near_vector(output.to_host(), { 3.0F });

    // When: A unit upstream gradient is backpropagated
    Tensor grad_output = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);
    Tensor grad_input = pool.backward(grad_output);
    synchronize_device();

    // Then: The gradient is routed only to the argmax location
    EXPECT_EQ(grad_input.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
    expect_near_vector(grad_input.to_host(), { 0.0F, 1.0F, 0.0F, 0.0F });
}

TEST_F(MaxPool2dTest, FourByFourWindows)
{
    // Given: A 2x2 pool over a 4x4 feature map
    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 1, 4, 4 },
        { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 8.0F, 7.0F, 6.0F, 5.0F,
            4.0F, 3.0F, 2.0F },
        Device::GPU);

    // When: The forward pass is computed
    Tensor output = pool.forward(input);
    synchronize_device();

    // Then: Each window keeps its maximum
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 1, 2, 2 }));
    expect_near_vector(output.to_host(), { 6.0F, 8.0F, 9.0F, 7.0F });

    // When: Unit gradients are backpropagated through the four windows
    Tensor grad_output = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F }, Device::GPU);
    Tensor grad_input = pool.backward(grad_output);
    synchronize_device();

    // Then: Gradients are 0 or 1 and sum to the number of windows
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
    // Given: Non-positive pooling hyperparameters
    // When: MaxPool2d is constructed
    // Then: Construction throws
    EXPECT_THROW(MaxPool2d(0, 2), std::runtime_error);
    EXPECT_THROW(MaxPool2d(2, 0), std::runtime_error);
    EXPECT_THROW(MaxPool2d(-1, 1), std::runtime_error);
}

TEST_F(MaxPool2dTest, ForwardRejectsCpuAndNonNchw)
{
    // Given: A 2x2 pool and illegal inputs
    MaxPool2d pool(2, 2);
    Tensor cpu = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::CPU);
    Tensor rank3 = Tensor::from_host({ 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: Forward is invoked
    // Then: Both inputs throw
    EXPECT_THROW(pool.forward(cpu), std::runtime_error);
    EXPECT_THROW(pool.forward(rank3), std::runtime_error);
}

TEST_F(MaxPool2dTest, BackwardWithoutForwardThrows)
{
    // Given: A pool that has not run forward
    MaxPool2d pool(2, 2);
    Tensor grad = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::GPU);

    // When: Backward is invoked
    // Then: The layer throws
    EXPECT_THROW(pool.backward(grad), std::runtime_error);
}

TEST_F(MaxPool2dTest, BackwardRejectsMismatchedGradientShape)
{
    // Given: A pooled 2x2 input
    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    (void)pool.forward(input);
    Tensor wrong_grad = Tensor::from_host({ 1, 1, 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);

    // When: Backward is given the unpooled shape
    // Then: The layer throws
    EXPECT_THROW(pool.backward(wrong_grad), std::runtime_error);
}

TEST_F(MaxPool2dTest, MultiChannelPoolsIndependently)
{
    // Given: Two channels with distinct maxima
    MaxPool2d pool(2, 2);
    Tensor input = Tensor::from_host({ 1, 2, 2, 2 }, { 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 5.0F }, Device::GPU);

    // When: The forward pass is computed
    Tensor output = pool.forward(input);
    synchronize_device();

    // Then: Each channel keeps its own maximum
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 2, 1, 1 }));
    expect_near_vector(output.to_host(), { 1.0F, 5.0F });
}

TEST_F(MaxPool2dTest, RepeatedForwardReusesDescriptors)
{
    // Given: A pool and two same-shaped inputs
    MaxPool2d pool(2, 2);
    Tensor first = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor second = Tensor::from_host({ 1, 1, 2, 2 }, { 4.0F, 3.0F, 2.0F, 1.0F }, Device::GPU);

    // When: Forward is run twice
    Tensor out1 = pool.forward(first);
    Tensor out2 = pool.forward(second);
    synchronize_device();

    // Then: Each call reports the window maximum
    expect_near_vector(out1.to_host(), { 4.0F });
    expect_near_vector(out2.to_host(), { 4.0F });
}
