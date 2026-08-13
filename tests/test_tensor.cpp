#include "DeepLearnLib/Tensor.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace dl;

namespace
{
constexpr float kEpsilon = 1e-5F;

auto has_cuda_device() -> bool
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        return false;
    }
    return count > 0;
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

auto synchronize_device() -> void
{
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
} // namespace

class TensorConstructorTest : public ::testing::Test
{
protected:
    bool HasCudaDevice()
    {
        return has_cuda_device();
    }
};

TEST_F(TensorConstructorTest, CpuAllocationAndZeroInitialization)
{
    std::vector<int> shape = { 2, 3, 4 };
    Tensor t(shape, Device::CPU);

    EXPECT_EQ(t.get_shape(), shape);
    EXPECT_EQ(t.get_size(), 24);
    EXPECT_EQ(t.get_device(), Device::CPU);

    const float* data_ptr = t.get_data();
    ASSERT_NE(data_ptr, nullptr);
    for (size_t i = 0; i < t.get_size(); ++i)
    {
        EXPECT_NEAR(data_ptr[i], 0.0F, kEpsilon);
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

    auto shared_memory = std::shared_ptr<float>(new float[4] { 1.0F, 2.0F, 3.0F, 4.0F }, CpuDeleter());

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

class GpuTensorTest : public ::testing::Test
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

TEST_F(GpuTensorTest, OneDimensionalAllocationShapeAndStrides)
{
    const std::vector<int> shape = { 8 };
    Tensor tensor(shape, Device::GPU);

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 8U);
    EXPECT_EQ(tensor.get_device(), Device::GPU);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int>{ 1 }));
    EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(GpuTensorTest, TwoDimensionalAllocationShapeAndStrides)
{
    const std::vector<int> shape = { 3, 4 };
    Tensor tensor(shape, Device::GPU);

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 12U);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int>{ 4, 1 }));
}

TEST_F(GpuTensorTest, ThreeDimensionalAllocationShapeAndStrides)
{
    const std::vector<int> shape = { 2, 3, 4 };
    Tensor tensor(shape, Device::GPU);

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 24U);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int>{ 12, 4, 1 }));
}

TEST_F(GpuTensorTest, NchwStridesMatchNetworkLayout)
{
    const std::vector<int> shape = { 1, 3, 8, 8 };
    Tensor tensor(shape, Device::GPU);

    EXPECT_EQ(tensor.get_size(), 192U);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int>{ 192, 64, 8, 1 }));
}

TEST_F(GpuTensorTest, FromHostVectorRoundTrip)
{
    const std::vector<int> shape = { 2, 3 };
    const std::vector<float> host_data = { 1.25F, -2.5F, 3.0F, 4.5F, 0.0F, 6.125F };

    Tensor tensor = Tensor::from_host(shape, host_data, Device::GPU);
    synchronize_device();

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_device(), Device::GPU);
    expect_near_vector(tensor.to_host(), host_data);
}

TEST_F(GpuTensorTest, FromHostPointerRoundTrip)
{
    const std::vector<int> shape = { 4 };
    const float host_data[] = { -1.0F, 2.0F, 3.5F, 8.25F };

    Tensor tensor = Tensor::from_host(shape, host_data, Device::GPU);
    synchronize_device();

    EXPECT_EQ(tensor.get_size(), 4U);
    expect_near_vector(tensor.to_host(), { -1.0F, 2.0F, 3.5F, 8.25F });
}

TEST_F(GpuTensorTest, FromHostCpuDeviceRoundTrip)
{
    const std::vector<int> shape = { 2, 2 };
    const std::vector<float> host_data = { 9.0F, 8.0F, 7.0F, 6.0F };

    Tensor tensor = Tensor::from_host(shape, host_data, Device::CPU);
    EXPECT_EQ(tensor.get_device(), Device::CPU);
    expect_near_vector(tensor.to_host(), host_data);
}

TEST_F(GpuTensorTest, FromHostRejectsMismatchedBufferSize)
{
    const std::vector<float> host_data = { 1.0F, 2.0F, 3.0F };
    EXPECT_THROW(Tensor::from_host({ 2, 2 }, host_data, Device::GPU), std::runtime_error);
}

TEST_F(GpuTensorTest, FromHostRejectsNullPointer)
{
    EXPECT_THROW(Tensor::from_host({ 2, 2 }, static_cast<const float*>(nullptr), Device::GPU), std::runtime_error);
}

TEST_F(GpuTensorTest, ZerosLikeMatchesShapeDeviceAndValues)
{
    Tensor source = Tensor::from_host({ 2, 3 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor zeros = Tensor::zeros_like(source);
    synchronize_device();

    EXPECT_EQ(zeros.get_shape(), source.get_shape());
    EXPECT_EQ(zeros.get_device(), Device::GPU);
    expect_near_vector(zeros.to_host(), std::vector<float>(6, 0.0F));
}

TEST_F(GpuTensorTest, SquareMatmul)
{
    Tensor lhs = Tensor::from_host({ 2, 2 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, std::vector<float>{ 5.0F, 6.0F, 7.0F, 8.0F }, Device::GPU);

    Tensor result = lhs.matmul(rhs);
    synchronize_device();

    EXPECT_EQ(result.get_shape(), (std::vector<int>{ 2, 2 }));
    expect_near_vector(result.to_host(), { 19.0F, 22.0F, 43.0F, 50.0F });
}

TEST_F(GpuTensorTest, RectangularMatmul)
{
    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 3, 4 },
                                   std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
                                                       11.0F, 12.0F },
                                   Device::GPU);

    Tensor result = lhs.matmul(rhs);
    synchronize_device();

    EXPECT_EQ(result.get_shape(), (std::vector<int>{ 2, 4 }));
    expect_near_vector(result.to_host(), { 38.0F, 44.0F, 50.0F, 56.0F, 78.0F, 92.0F, 106.0F, 120.0F });
}

TEST_F(GpuTensorTest, IdentityMatmulLeavesMatrixUnchanged)
{
    Tensor matrix = Tensor::from_host({ 3, 3 },
                                      std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F },
                                      Device::GPU);
    Tensor identity = Tensor::from_host({ 3, 3 },
                                        std::vector<float>{ 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F },
                                        Device::GPU);

    Tensor result = matrix.matmul(identity);
    synchronize_device();
    expect_near_vector(result.to_host(), matrix.to_host());
}

TEST_F(GpuTensorTest, MatmulMismatchedInnerDimensionsThrows)
{
    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 4, 5 }, std::vector<float>(20, 1.0F), Device::GPU);

    EXPECT_THROW(lhs.matmul(rhs), std::runtime_error);
}

TEST_F(GpuTensorTest, ElementwiseTensorAddition)
{
    Tensor lhs = Tensor::from_host({ 2, 2 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, std::vector<float>{ 10.0F, 20.0F, 30.0F, 40.0F }, Device::GPU);

    Tensor result = lhs + rhs;
    synchronize_device();
    expect_near_vector(result.to_host(), { 11.0F, 22.0F, 33.0F, 44.0F });
}

TEST_F(GpuTensorTest, ElementwiseTensorSubtraction)
{
    Tensor lhs = Tensor::from_host({ 4 }, std::vector<float>{ 5.0F, 4.0F, 3.0F, 2.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 4 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 8.0F }, Device::GPU);

    Tensor result = lhs - rhs;
    synchronize_device();
    expect_near_vector(result.to_host(), { 4.0F, 2.0F, 0.0F, -6.0F });
}

TEST_F(GpuTensorTest, ElementwiseTensorMultiplication)
{
    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float>{ 1.0F, -2.0F, 3.0F, 4.0F, 0.5F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 3 }, std::vector<float>{ 2.0F, 3.0F, 4.0F, 0.0F, 8.0F, -1.0F }, Device::GPU);

    Tensor result = lhs * rhs;
    synchronize_device();
    expect_near_vector(result.to_host(), { 2.0F, -6.0F, 12.0F, 0.0F, 4.0F, -6.0F });
}

TEST_F(GpuTensorTest, TensorScalarMultiplicationAndAddition)
{
    Tensor tensor = Tensor::from_host({ 3 }, std::vector<float>{ 1.0F, -2.0F, 4.0F }, Device::GPU);

    Tensor scaled = tensor * 2.5F;
    Tensor shifted = tensor + 1.0F;
    synchronize_device();

    expect_near_vector(scaled.to_host(), { 2.5F, -5.0F, 10.0F });
    expect_near_vector(shifted.to_host(), { 2.0F, -1.0F, 5.0F });
}

TEST_F(GpuTensorTest, ElementwiseSizeMismatchThrows)
{
    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    EXPECT_THROW(lhs + rhs, std::runtime_error);
    EXPECT_THROW(lhs - rhs, std::runtime_error);
    EXPECT_THROW(lhs * rhs, std::runtime_error);
}

TEST_F(GpuTensorTest, ElementwiseCpuOperandThrows)
{
    Tensor gpu = Tensor::from_host({ 2 }, std::vector<float>{ 1.0F, 2.0F }, Device::GPU);
    Tensor cpu = Tensor::from_host({ 2 }, std::vector<float>{ 3.0F, 4.0F }, Device::CPU);

    EXPECT_THROW(gpu + cpu, std::runtime_error);
    EXPECT_THROW(cpu * 2.0F, std::runtime_error);
}

TEST_F(GpuTensorTest, ClampBoundsValues)
{
    Tensor tensor = Tensor::from_host({ 5 }, std::vector<float>{ -4.0F, -0.5F, 0.0F, 0.7F, 3.0F }, Device::GPU);
    Tensor clamped = tensor.clamp(-1.0F, 1.0F);
    synchronize_device();
    expect_near_vector(clamped.to_host(), { -1.0F, -0.5F, 0.0F, 0.7F, 1.0F });
}

TEST_F(GpuTensorTest, ClampInvalidRangeThrows)
{
    Tensor tensor = Tensor::from_host({ 2 }, std::vector<float>{ 1.0F, 2.0F }, Device::GPU);
    EXPECT_THROW(tensor.clamp(1.0F, 0.0F), std::runtime_error);
}

TEST_F(GpuTensorTest, GlobalSumPositiveValues)
{
    Tensor tensor = Tensor::from_host({ 2, 2 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor reduced = tensor.sum();
    synchronize_device();

    EXPECT_EQ(reduced.get_shape(), (std::vector<int>{ 1 }));
    expect_near_vector(reduced.to_host(), { 10.0F });
}

TEST_F(GpuTensorTest, GlobalSumNegativeAndMixedValues)
{
    Tensor tensor = Tensor::from_host({ 4 }, std::vector<float>{ 1.0F, -2.0F, 3.0F, -4.5F }, Device::GPU);
    Tensor reduced = tensor.sum(-1);
    synchronize_device();
    expect_near_vector(reduced.to_host(), { -2.5F });
}

TEST_F(GpuTensorTest, SumAlongAxisThrows)
{
    Tensor tensor = Tensor::from_host({ 3 }, std::vector<float>{ 1.0F, 2.0F, 3.0F }, Device::GPU);
    EXPECT_THROW(tensor.sum(0), std::runtime_error);
}

TEST_F(GpuTensorTest, ViewReshapeSharesStorageAndValues)
{
    Tensor tensor = Tensor::from_host({ 2, 3 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor viewed = tensor.view({ 3, 2 });

    EXPECT_EQ(viewed.get_shape(), (std::vector<int>{ 3, 2 }));
    EXPECT_EQ(viewed.get_strides(), (std::vector<int>{ 2, 1 }));
    EXPECT_EQ(viewed.get_size(), tensor.get_size());
    EXPECT_EQ(viewed.data(), tensor.data());
    expect_near_vector(viewed.to_host(), tensor.to_host());
}

TEST_F(GpuTensorTest, ViewInfersMinusOneDimension)
{
    Tensor tensor = Tensor::from_host({ 2, 4 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F },
                                      Device::GPU);
    Tensor viewed = tensor.view({ -1, 2 });

    EXPECT_EQ(viewed.get_shape(), (std::vector<int>{ 4, 2 }));
    expect_near_vector(viewed.to_host(), tensor.to_host());
}

TEST_F(GpuTensorTest, ViewIncompatibleShapeThrows)
{
    Tensor tensor = Tensor::from_host({ 2, 3 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    EXPECT_THROW(tensor.view({ 2, 2 }), std::runtime_error);
    EXPECT_THROW(tensor.view({ -1, -1 }), std::runtime_error);
}

TEST_F(GpuTensorTest, TransposeTwoDimensional)
{
    Tensor tensor = Tensor::from_host({ 2, 3 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor transposed = tensor.transpose();
    synchronize_device();

    EXPECT_EQ(transposed.get_shape(), (std::vector<int>{ 3, 2 }));
    expect_near_vector(transposed.to_host(), { 1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F });
}

TEST_F(GpuTensorTest, TransposeThenMatmulMatchesOriginalProductLayout)
{
    Tensor matrix = Tensor::from_host({ 2, 2 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor transposed = matrix.transpose();
    synchronize_device();
    expect_near_vector(transposed.to_host(), { 1.0F, 3.0F, 2.0F, 4.0F });
}

TEST_F(GpuTensorTest, TransposeRejectsNonTwoDimensionalTensors)
{
    Tensor vector = Tensor::from_host({ 4 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor volume = Tensor::from_host({ 2, 2, 1 }, std::vector<float>{ 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    EXPECT_THROW(vector.transpose(), std::runtime_error);
    EXPECT_THROW(volume.transpose(), std::runtime_error);
}
