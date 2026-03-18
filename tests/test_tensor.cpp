#include "DeepLearnLib/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>

using namespace dl;

class TensorConstructorTest : public ::testing::Test
{
protected:
    bool HasCudaDevice()
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess)
            return false;
        return count > 0;
    }
};
TEST_F(TensorConstructorTest, CpuAllocationAndZeroInitialization)
{
    std::vector<int> shape = { 2, 3, 4 };
    Tensor t(shape, Device::CPU);

    EXPECT_EQ(t.get_shape(), shape);
    EXPECT_EQ(t.get_size(), 24); // 2 * 3 * 4 = 24
    EXPECT_EQ(t.get_device(), Device::CPU);

    const float* data_ptr = t.get_data();
    ASSERT_NE(data_ptr, nullptr);
    for (size_t i = 0; i < t.get_size(); ++i)
    {
        EXPECT_FLOAT_EQ(data_ptr[i], 0.0f);
    }
}

TEST_F(TensorConstructorTest, ScalarCpuAllocation)
{
    std::vector<int> empty_shape = {};
    Tensor t(empty_shape, Device::CPU);

    EXPECT_EQ(t.get_size(), 1);
    EXPECT_EQ(t.get_shape(), empty_shape);
    EXPECT_NE(t.get_data(), nullptr);
}

TEST_F(TensorConstructorTest, GpuAllocationBehavior)
{
    std::vector<int> shape = { 10, 10 };

    if (HasCudaDevice())
    {
        EXPECT_NO_THROW({
            Tensor t(shape, Device::GPU);
            EXPECT_EQ(t.get_size(), 100);
            EXPECT_EQ(t.get_device(), Device::GPU);
            EXPECT_NE(t.get_data(), nullptr);
        });
    }
    else
    {
        EXPECT_THROW({ Tensor t(shape, Device::GPU); }, std::runtime_error);
    }
}

TEST_F(TensorConstructorTest, ViewConstructorSharesMemoryOwnership)
{
    std::vector<int> shape = { 2, 2 };
    std::vector<int> strides = { 2, 1 };

    auto shared_memory = std::shared_ptr<float>(new float[4] { 1.0f, 2.0f, 3.0f, 4.0f }, CpuDeleter());

    ASSERT_EQ(shared_memory.use_count(), 1);

    {
        Tensor view(shape, strides, shared_memory, Device::CPU);

        EXPECT_EQ(view.get_shape(), shape);
        EXPECT_EQ(view.get_strides(), strides);
        EXPECT_EQ(view.get_size(), 4);
        EXPECT_EQ(view.get_device(), Device::CPU);

        EXPECT_EQ(view.get_data(), shared_memory.get());

        EXPECT_EQ(shared_memory.use_count(), 2);
    }

    EXPECT_EQ(shared_memory.use_count(), 1);
}
