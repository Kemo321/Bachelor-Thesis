#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace dl;
using namespace dllib_test;

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

    Tensor tensor(shape, Device::CPU);

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 24);
    EXPECT_EQ(tensor.get_device(), Device::CPU);
    const float* data_ptr = tensor.get_data();
    ASSERT_NE(data_ptr, nullptr);
    for (size_t i = 0; i < tensor.get_size(); ++i)
    {
        EXPECT_NEAR(data_ptr[i], 0.0F, kTensorEpsilon);
    }
}

TEST_F(TensorConstructorTest, ScalarCpuAllocation)
{

    std::vector<int> empty_shape = {};

    Tensor tensor(empty_shape, Device::CPU);

    EXPECT_EQ(tensor.get_size(), 1);
    EXPECT_EQ(tensor.get_shape(), empty_shape);
    EXPECT_NE(tensor.get_data(), nullptr);
}

TEST_F(TensorConstructorTest, GpuAllocationBehavior)
{

    std::vector<int> shape = { 10, 10 };

    if (HasCudaDevice())
    {
        EXPECT_NO_THROW({
            Tensor tensor(shape, Device::GPU);
            EXPECT_EQ(tensor.get_size(), 100);
            EXPECT_EQ(tensor.get_device(), Device::GPU);
            EXPECT_NE(tensor.get_data(), nullptr);
        });
    }
    else
    {
        EXPECT_THROW({ Tensor tensor(shape, Device::GPU); }, std::runtime_error);
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

class GpuTensorTest : public GpuTest
{
};

TEST_F(GpuTensorTest, OneDimensionalAllocationShapeAndStrides)
{

    const std::vector<int> shape = { 8 };

    Tensor tensor(shape, Device::GPU);

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 8U);
    EXPECT_EQ(tensor.get_device(), Device::GPU);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int> { 1 }));
    EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(GpuTensorTest, TwoDimensionalAllocationShapeAndStrides)
{

    const std::vector<int> shape = { 3, 4 };

    Tensor tensor(shape, Device::GPU);

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 12U);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int> { 4, 1 }));
}

TEST_F(GpuTensorTest, ThreeDimensionalAllocationShapeAndStrides)
{

    const std::vector<int> shape = { 2, 3, 4 };

    Tensor tensor(shape, Device::GPU);

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 24U);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int> { 12, 4, 1 }));
}

TEST_F(GpuTensorTest, NchwStridesMatchNetworkLayout)
{

    const std::vector<int> shape = { 1, 3, 8, 8 };

    Tensor tensor(shape, Device::GPU);

    EXPECT_EQ(tensor.get_size(), 192U);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int> { 192, 64, 8, 1 }));
}

TEST_F(GpuTensorTest, FromHostVectorRoundTrip)
{

    const std::vector<int> shape = { 2, 3 };
    const std::vector<float> host_data = { 1.25F, -2.5F, 3.0F, 4.5F, 0.0F, 6.125F };

    Tensor tensor = Tensor::from_host(shape, host_data, Device::GPU);
    synchronize_device();

    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_device(), Device::GPU);
    expect_near_vector(tensor.to_host(), host_data, kTensorEpsilon);
}

TEST_F(GpuTensorTest, FromHostPointerRoundTrip)
{

    const std::vector<int> shape = { 4 };
    const float host_data[] = { -1.0F, 2.0F, 3.5F, 8.25F };

    Tensor tensor = Tensor::from_host(shape, host_data, Device::GPU);
    synchronize_device();

    EXPECT_EQ(tensor.get_size(), 4U);
    expect_near_vector(tensor.to_host(), { -1.0F, 2.0F, 3.5F, 8.25F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, FromHostCpuDeviceRoundTrip)
{

    const std::vector<int> shape = { 2, 2 };
    const std::vector<float> host_data = { 9.0F, 8.0F, 7.0F, 6.0F };

    Tensor tensor = Tensor::from_host(shape, host_data, Device::CPU);

    EXPECT_EQ(tensor.get_device(), Device::CPU);
    expect_near_vector(tensor.to_host(), host_data, kTensorEpsilon);
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

    Tensor source = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    Tensor zeros = Tensor::zeros_like(source);
    synchronize_device();

    EXPECT_EQ(zeros.get_shape(), source.get_shape());
    EXPECT_EQ(zeros.get_device(), Device::GPU);
    expect_near_vector(zeros.to_host(), std::vector<float>(6, 0.0F), kTensorEpsilon);
}

TEST_F(GpuTensorTest, SquareMatmul)
{

    Tensor lhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 5.0F, 6.0F, 7.0F, 8.0F }, Device::GPU);

    Tensor result = lhs.matmul(rhs);
    synchronize_device();

    EXPECT_EQ(result.get_shape(), (std::vector<int> { 2, 2 }));
    expect_near_vector(result.to_host(), { 19.0F, 22.0F, 43.0F, 50.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, RectangularMatmul)
{

    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 3, 4 },
        std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
            11.0F, 12.0F },
        Device::GPU);

    Tensor result = lhs.matmul(rhs);
    synchronize_device();

    EXPECT_EQ(result.get_shape(), (std::vector<int> { 2, 4 }));
    expect_near_vector(result.to_host(), { 38.0F, 44.0F, 50.0F, 56.0F, 83.0F, 98.0F, 113.0F, 128.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, LogicalTransposeMatmulMatchesPhysicalTranspose)
{

    Tensor lhs = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 4 },
        { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F }, Device::GPU);

    Tensor logical = lhs.matmul(rhs, true, false);
    Tensor physical = lhs.transpose().matmul(rhs);
    synchronize_device();

    EXPECT_EQ(logical.get_shape(), (std::vector<int> { 3, 4 }));
    expect_near_vector(logical.to_host(), physical.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, LogicalTransposeBMatmulMatchesPhysicalTranspose)
{

    Tensor lhs = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 4, 3 },
        { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F }, Device::GPU);

    Tensor logical = lhs.matmul(rhs, false, true);
    Tensor physical = lhs.matmul(rhs.transpose());
    synchronize_device();

    EXPECT_EQ(logical.get_shape(), (std::vector<int> { 2, 4 }));
    expect_near_vector(logical.to_host(), physical.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, InPlaceAddMulAndAddScaledDoNotAllocateTemps)
{

    Tensor lhs = Tensor::from_host({ 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, { 10.0F, 20.0F, 30.0F, 40.0F }, Device::GPU);
    const float* lhs_ptr = lhs.data();

    lhs.add_(rhs);
    lhs.mul_(0.5F);
    lhs.add_scaled_(rhs, 0.1F);
    synchronize_device();

    EXPECT_EQ(lhs.data(), lhs_ptr);
    expect_near_vector(lhs.to_host(), { 6.5F, 13.0F, 19.5F, 26.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, MatmulIntoAccumulatesIntoExistingBuffer)
{

    Tensor lhs = Tensor::from_host({ 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, { 5.0F, 6.0F, 7.0F, 8.0F }, Device::GPU);
    Tensor out = Tensor::from_host({ 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F }, Device::GPU);
    const float* out_ptr = out.data();

    lhs.matmul_into(rhs, out, false, false, 1.0F);
    synchronize_device();

    EXPECT_EQ(out.data(), out_ptr);
    expect_near_vector(out.to_host(), { 20.0F, 23.0F, 44.0F, 51.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, AddRowBroadcastsBiasAcrossBatch)
{

    Tensor rows = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor bias = Tensor::from_host({ 1, 3 }, { 0.5F, -1.0F, 2.0F }, Device::GPU);

    rows.add_row_(bias);
    synchronize_device();

    expect_near_vector(rows.to_host(), { 1.5F, 1.0F, 5.0F, 4.5F, 4.0F, 8.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, AddSumRowsAccumulatesWithBeta)
{

    Tensor matrix = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor acc = Tensor::from_host({ 1, 3 }, { 10.0F, 10.0F, 10.0F }, Device::GPU);

    acc.add_sum_rows_(matrix, 0.5F);
    synchronize_device();

    expect_near_vector(acc.to_host(), { 10.0F, 12.0F, 14.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, IdentityMatmulLeavesMatrixUnchanged)
{

    Tensor matrix = Tensor::from_host({ 3, 3 },
        std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F },
        Device::GPU);
    Tensor identity = Tensor::from_host({ 3, 3 },
        std::vector<float> { 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F },
        Device::GPU);

    Tensor result = matrix.matmul(identity);
    synchronize_device();

    expect_near_vector(result.to_host(), matrix.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, MatmulMismatchedInnerDimensionsThrows)
{

    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 4, 5 }, std::vector<float>(20, 1.0F), Device::GPU);

    EXPECT_THROW(lhs.matmul(rhs), std::runtime_error);
}

TEST_F(GpuTensorTest, ElementwiseTensorAddition)
{

    Tensor lhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 10.0F, 20.0F, 30.0F, 40.0F }, Device::GPU);

    Tensor result = lhs + rhs;
    synchronize_device();

    expect_near_vector(result.to_host(), { 11.0F, 22.0F, 33.0F, 44.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ElementwiseTensorSubtraction)
{

    Tensor lhs = Tensor::from_host({ 4 }, std::vector<float> { 5.0F, 4.0F, 3.0F, 2.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 4 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 8.0F }, Device::GPU);

    Tensor result = lhs - rhs;
    synchronize_device();

    expect_near_vector(result.to_host(), { 4.0F, 2.0F, 0.0F, -6.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ElementwiseTensorMultiplication)
{

    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, -2.0F, 3.0F, 4.0F, 0.5F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 2.0F, 3.0F, 4.0F, 0.0F, 8.0F, -1.0F }, Device::GPU);

    Tensor result = lhs * rhs;
    synchronize_device();

    expect_near_vector(result.to_host(), { 2.0F, -6.0F, 12.0F, 0.0F, 4.0F, -6.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, TensorScalarMultiplicationAndAddition)
{

    Tensor tensor = Tensor::from_host({ 3 }, std::vector<float> { 1.0F, -2.0F, 4.0F }, Device::GPU);

    Tensor scaled = tensor * 2.5F;
    Tensor shifted = tensor + 1.0F;
    synchronize_device();

    expect_near_vector(scaled.to_host(), { 2.5F, -5.0F, 10.0F }, kTensorEpsilon);
    expect_near_vector(shifted.to_host(), { 2.0F, -1.0F, 5.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ElementwiseSizeMismatchThrows)
{

    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    EXPECT_THROW(lhs + rhs, std::runtime_error);
    EXPECT_THROW(lhs - rhs, std::runtime_error);
    EXPECT_THROW(lhs * rhs, std::runtime_error);
}

TEST_F(GpuTensorTest, ElementwiseCpuOperandThrows)
{

    Tensor gpu = Tensor::from_host({ 2 }, std::vector<float> { 1.0F, 2.0F }, Device::GPU);
    Tensor cpu = Tensor::from_host({ 2 }, std::vector<float> { 3.0F, 4.0F }, Device::CPU);

    EXPECT_THROW(gpu + cpu, std::runtime_error);
    EXPECT_THROW(cpu * 2.0F, std::runtime_error);
}

TEST_F(GpuTensorTest, ClampBoundsValues)
{

    Tensor tensor = Tensor::from_host({ 5 }, std::vector<float> { -4.0F, -0.5F, 0.0F, 0.7F, 3.0F }, Device::GPU);

    Tensor clamped = tensor.clamp(-1.0F, 1.0F);
    synchronize_device();

    expect_near_vector(clamped.to_host(), { -1.0F, -0.5F, 0.0F, 0.7F, 1.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ClampInvalidRangeThrows)
{

    Tensor tensor = Tensor::from_host({ 2 }, std::vector<float> { 1.0F, 2.0F }, Device::GPU);

    EXPECT_THROW(tensor.clamp(1.0F, 0.0F), std::runtime_error);
}

TEST_F(GpuTensorTest, GlobalSumPositiveValues)
{

    Tensor tensor = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    Tensor reduced = tensor.sum();
    synchronize_device();

    EXPECT_EQ(reduced.get_shape(), (std::vector<int> { 1 }));
    expect_near_vector(reduced.to_host(), { 10.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, GlobalSumNegativeAndMixedValues)
{

    Tensor tensor = Tensor::from_host({ 4 }, std::vector<float> { 1.0F, -2.0F, 3.0F, -4.5F }, Device::GPU);

    Tensor reduced = tensor.sum(-1);
    synchronize_device();

    expect_near_vector(reduced.to_host(), { -2.5F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, SumAlongAxisThrows)
{

    Tensor tensor = Tensor::from_host({ 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F }, Device::GPU);

    EXPECT_THROW(tensor.sum(0), std::runtime_error);
}

TEST_F(GpuTensorTest, ViewReshapeSharesStorageAndValues)
{

    Tensor tensor = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    Tensor viewed = tensor.view({ 3, 2 });

    EXPECT_EQ(viewed.get_shape(), (std::vector<int> { 3, 2 }));
    EXPECT_EQ(viewed.get_strides(), (std::vector<int> { 2, 1 }));
    EXPECT_EQ(viewed.get_size(), tensor.get_size());
    EXPECT_EQ(viewed.data(), tensor.data());
    expect_near_vector(viewed.to_host(), tensor.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, ViewInfersMinusOneDimension)
{

    Tensor tensor = Tensor::from_host({ 2, 4 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F },
        Device::GPU);

    Tensor viewed = tensor.view({ -1, 2 });

    EXPECT_EQ(viewed.get_shape(), (std::vector<int> { 4, 2 }));
    expect_near_vector(viewed.to_host(), tensor.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, ViewIncompatibleShapeThrows)
{

    Tensor tensor = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    EXPECT_THROW(tensor.view({ 2, 2 }), std::runtime_error);
    EXPECT_THROW(tensor.view({ -1, -1 }), std::runtime_error);
}

TEST_F(GpuTensorTest, TransposeTwoDimensional)
{

    Tensor tensor = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    Tensor transposed = tensor.transpose();
    synchronize_device();

    EXPECT_EQ(transposed.get_shape(), (std::vector<int> { 3, 2 }));
    expect_near_vector(transposed.to_host(), { 1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, TransposeThenMatmulMatchesOriginalProductLayout)
{

    Tensor matrix = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    Tensor transposed = matrix.transpose();
    synchronize_device();

    expect_near_vector(transposed.to_host(), { 1.0F, 3.0F, 2.0F, 4.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, TransposeRejectsNonTwoDimensionalTensors)
{

    Tensor vector = Tensor::from_host({ 4 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor volume = Tensor::from_host({ 2, 2, 1 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    EXPECT_THROW(vector.transpose(), std::runtime_error);
    EXPECT_THROW(volume.transpose(), std::runtime_error);
}

TEST_F(GpuTensorTest, VectorMatrixMatmul)
{

    Tensor vector = Tensor::from_host({ 3 }, { 1.0F, 2.0F, 3.0F }, Device::GPU);
    Tensor matrix = Tensor::from_host({ 3, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F }, Device::GPU);

    Tensor result = vector.matmul(matrix);
    synchronize_device();

    EXPECT_EQ(result.get_shape(), (std::vector<int> { 2 }));
    expect_near_vector(result.to_host(), { 4.0F, 5.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, MatmulOnCpuThrows)
{

    Tensor lhs = Tensor::from_host({ 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F }, Device::CPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::CPU);

    EXPECT_THROW(lhs.matmul(rhs), std::runtime_error);
}

TEST_F(GpuTensorTest, ChainedElementwiseOps)
{

    Tensor a = Tensor::from_host({ 3 }, { 1.0F, 2.0F, 3.0F }, Device::GPU);
    Tensor b = Tensor::from_host({ 3 }, { 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor c = Tensor::from_host({ 3 }, { 2.0F, 2.0F, 2.0F }, Device::GPU);

    Tensor result = (a + b) * c;
    synchronize_device();

    expect_near_vector(result.to_host(), { 10.0F, 14.0F, 18.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ClampEqualBoundsSaturates)
{

    Tensor tensor = Tensor::from_host({ 3 }, { 0.0F, 2.0F, 5.0F }, Device::GPU);

    Tensor clamped = tensor.clamp(2.0F, 2.0F);
    synchronize_device();

    expect_near_vector(clamped.to_host(), { 2.0F, 2.0F, 2.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ViewFlattensWithMinusOne)
{

    Tensor tensor = Tensor::from_host({ 2, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F }, Device::GPU);

    Tensor viewed = tensor.view({ -1 });

    EXPECT_EQ(viewed.get_shape(), (std::vector<int> { 8 }));
    expect_near_vector(viewed.to_host(), tensor.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, ViewRejectsInvalidNegativeDimension)
{

    Tensor tensor = Tensor::from_host({ 4 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    EXPECT_THROW(tensor.view({ -2 }), std::runtime_error);
}

TEST_F(GpuTensorTest, ViewIdentityKeepsShape)
{

    Tensor tensor = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    Tensor viewed = tensor.view({ 2, 3 });

    EXPECT_EQ(viewed.data(), tensor.data());
    EXPECT_EQ(viewed.get_shape(), tensor.get_shape());
    expect_near_vector(viewed.to_host(), tensor.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, SumOfZerosIsZero)
{

    Tensor tensor = Tensor::from_host({ 3, 3 }, std::vector<float>(9, 0.0F), Device::GPU);

    Tensor reduced = tensor.sum();
    synchronize_device();

    expect_near_vector(reduced.to_host(), { 0.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, TransposeOneByN)
{

    Tensor row = Tensor::from_host({ 1, 4 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    Tensor column = row.transpose();
    synchronize_device();

    EXPECT_EQ(column.get_shape(), (std::vector<int> { 4, 1 }));
    expect_near_vector(column.to_host(), { 1.0F, 2.0F, 3.0F, 4.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, MoveConstructorTransfersOwnership)
{

    Tensor original = Tensor::from_host({ 2 }, { 7.0F, 8.0F }, Device::GPU);
    const float* pointer = original.data();

    Tensor moved(std::move(original));
    synchronize_device();

    EXPECT_EQ(moved.data(), pointer);
    expect_near_vector(moved.to_host(), { 7.0F, 8.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ZerosLikeCpuSourceStaysOnCpu)
{

    Tensor source = Tensor::from_host({ 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::CPU);

    Tensor zeros = Tensor::zeros_like(source);

    EXPECT_EQ(zeros.get_device(), Device::CPU);
    EXPECT_EQ(zeros.get_shape(), source.get_shape());
    expect_near_vector(zeros.to_host(), std::vector<float>(4, 0.0F), kTensorEpsilon);
}

TEST_F(GpuTensorTest, CpuToHostRoundTrip)
{

    Tensor tensor = Tensor::from_host({ 3 }, { 1.5F, 2.5F, 3.5F }, Device::CPU);

    const std::vector<float> host = tensor.to_host();

    expect_near_vector(host, { 1.5F, 2.5F, 3.5F }, kTensorEpsilon);
}
