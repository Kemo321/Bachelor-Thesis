#include "test_helpers.hpp"

#include "DeepLearnLib/Flatten.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <vector>

using namespace dl;
using namespace dllib_test;

class FlattenTest : public GpuTest
{
};

TEST_F(FlattenTest, ReshapesAndUnflattensGradient)
{
    // Given: A Flatten layer and a 2x3x2x2 tensor with sequential values
    Flatten flatten;
    const std::vector<int> input_shape = { 2, 3, 2, 2 };
    std::vector<float> host_input(24);
    for (int index = 0; index < 24; ++index)
    {
        host_input[static_cast<size_t>(index)] = static_cast<float>(index);
    }
    Tensor input = Tensor::from_host(input_shape, host_input, Device::GPU);

    // When: The forward pass is computed
    Tensor output = flatten.forward(input);
    synchronize_device();

    // Then: The batch is preserved and remaining dims collapse to 12
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 2, 12 }));
    expect_near_vector(output.to_host(), host_input);

    // When: The same values are backpropagated
    Tensor grad_output = Tensor::from_host({ 2, 12 }, host_input, Device::GPU);
    Tensor grad_input = flatten.backward(grad_output);
    synchronize_device();

    // Then: The gradient is restored to the original NCHW shape
    EXPECT_EQ(grad_input.get_shape(), input_shape);
    expect_near_vector(grad_input.to_host(), host_input);
}

TEST_F(FlattenTest, AlreadyRankTwoIsUnchanged)
{
    // Given: A rank-2 [batch, features] tensor
    Flatten flatten;
    const std::vector<float> host = { 1.0F, 2.0F, 3.0F, 4.0F };
    Tensor input = Tensor::from_host({ 2, 2 }, host, Device::GPU);

    // When: The forward pass is computed
    Tensor output = flatten.forward(input);
    synchronize_device();

    // Then: Shape and values are unchanged
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 2, 2 }));
    expect_near_vector(output.to_host(), host);
}

TEST_F(FlattenTest, FiveDimensionalInput)
{
    // Given: A 2x2x2x2x2 tensor
    Flatten flatten;
    std::vector<float> host(32);
    for (int index = 0; index < 32; ++index)
    {
        host[static_cast<size_t>(index)] = static_cast<float>(index);
    }
    Tensor input = Tensor::from_host({ 2, 2, 2, 2, 2 }, host, Device::GPU);

    // When: The forward pass is computed
    Tensor output = flatten.forward(input);
    synchronize_device();

    // Then: The result is [2, 16]
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 2, 16 }));
    expect_near_vector(output.to_host(), host);
}

TEST_F(FlattenTest, ScalarAndCpuInputsThrow)
{
    // Given: A scalar GPU tensor and a CPU NCHW tensor
    Flatten flatten;
    Tensor scalar(std::vector<int> {}, Device::GPU);
    Tensor cpu = Tensor::from_host({ 1, 1, 1, 1 }, { 1.0F }, Device::CPU);

    // When: Forward is invoked
    // Then: Both inputs throw
    EXPECT_THROW(flatten.forward(scalar), std::runtime_error);
    EXPECT_THROW(flatten.forward(cpu), std::runtime_error);
}

TEST_F(FlattenTest, BackwardWithoutForwardThrows)
{
    // Given: A Flatten layer that has not run forward
    Flatten flatten;
    Tensor grad = Tensor::from_host({ 1, 4 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: Backward is invoked
    // Then: The layer throws
    EXPECT_THROW(flatten.backward(grad), std::runtime_error);
}

TEST_F(FlattenTest, OneDimensionalTreatsLengthAsBatch)
{
    // Given: A length-4 vector (batch = 4, features = 1)
    Flatten flatten;
    Tensor input = Tensor::from_host({ 4 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: The forward pass is computed
    Tensor output = flatten.forward(input);
    synchronize_device();

    // Then: The result is [4, 1]
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 4, 1 }));
    expect_near_vector(output.to_host(), { 1.0F, 2.0F, 3.0F, 4.0F });
}
