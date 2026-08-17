#include "test_helpers.hpp"

#include "DeepLearnLib/Tensor.hpp"

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
    // Given: A 2x3x4 CPU tensor shape
    std::vector<int> shape = { 2, 3, 4 };

    // When: The tensor is allocated on CPU
    Tensor t(shape, Device::CPU);

    // Then: Size, device, and zero-filled storage match the request
    EXPECT_EQ(t.get_shape(), shape);
    EXPECT_EQ(t.get_size(), 24);
    EXPECT_EQ(t.get_device(), Device::CPU);
    const float* data_ptr = t.get_data();
    ASSERT_NE(data_ptr, nullptr);
    for (size_t i = 0; i < t.get_size(); ++i)
    {
        EXPECT_NEAR(data_ptr[i], 0.0F, kTensorEpsilon);
    }
}

TEST_F(TensorConstructorTest, ScalarCpuAllocation)
{
    // Given: An empty shape representing a scalar
    std::vector<int> empty_shape = {};

    // When: The tensor is allocated on CPU
    Tensor t(empty_shape, Device::CPU);

    // Then: The tensor holds a single element
    EXPECT_EQ(t.get_size(), 1);
    EXPECT_EQ(t.get_shape(), empty_shape);
    EXPECT_NE(t.get_data(), nullptr);
}

TEST_F(TensorConstructorTest, GpuAllocationBehavior)
{
    // Given: A 10x10 shape and the current CUDA device availability
    std::vector<int> shape = { 10, 10 };

    // When: A GPU tensor is constructed
    // Then: Construction succeeds on a CUDA device, otherwise it throws
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
    // Given: Host memory owned by a shared_ptr
    std::vector<int> shape = { 2, 2 };
    std::vector<int> strides = { 2, 1 };
    auto shared_memory = std::shared_ptr<float>(new float[4] { 1.0F, 2.0F, 3.0F, 4.0F }, CpuDeleter());
    ASSERT_EQ(shared_memory.use_count(), 1);

    {
        // When: A view tensor is constructed over that storage
        Tensor view(shape, strides, shared_memory, Device::CPU);

        // Then: The view aliases the same pointer and bumps the refcount
        EXPECT_EQ(view.get_shape(), shape);
        EXPECT_EQ(view.get_strides(), strides);
        EXPECT_EQ(view.get_size(), 4);
        EXPECT_EQ(view.get_device(), Device::CPU);
        EXPECT_EQ(view.get_data(), shared_memory.get());
        EXPECT_EQ(shared_memory.use_count(), 2);
    }

    // Then: Destroying the view releases the extra reference
    EXPECT_EQ(shared_memory.use_count(), 1);
}

class GpuTensorTest : public GpuTest
{
};

TEST_F(GpuTensorTest, OneDimensionalAllocationShapeAndStrides)
{
    // Given: A length-8 GPU tensor shape
    const std::vector<int> shape = { 8 };

    // When: The tensor is allocated on GPU
    Tensor tensor(shape, Device::GPU);

    // Then: Shape, size, device, and unit stride are set
    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 8U);
    EXPECT_EQ(tensor.get_device(), Device::GPU);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int> { 1 }));
    EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(GpuTensorTest, TwoDimensionalAllocationShapeAndStrides)
{
    // Given: A 3x4 GPU tensor shape
    const std::vector<int> shape = { 3, 4 };

    // When: The tensor is allocated on GPU
    Tensor tensor(shape, Device::GPU);

    // Then: Row-major strides are {4, 1}
    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 12U);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int> { 4, 1 }));
}

TEST_F(GpuTensorTest, ThreeDimensionalAllocationShapeAndStrides)
{
    // Given: A 2x3x4 GPU tensor shape
    const std::vector<int> shape = { 2, 3, 4 };

    // When: The tensor is allocated on GPU
    Tensor tensor(shape, Device::GPU);

    // Then: Contiguous 3D strides are {12, 4, 1}
    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_size(), 24U);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int> { 12, 4, 1 }));
}

TEST_F(GpuTensorTest, NchwStridesMatchNetworkLayout)
{
    // Given: An NCHW 1x3x8x8 tensor
    const std::vector<int> shape = { 1, 3, 8, 8 };

    // When: The tensor is allocated on GPU
    Tensor tensor(shape, Device::GPU);

    // Then: Strides match NCHW layout
    EXPECT_EQ(tensor.get_size(), 192U);
    EXPECT_EQ(tensor.get_strides(), (std::vector<int> { 192, 64, 8, 1 }));
}

TEST_F(GpuTensorTest, FromHostVectorRoundTrip)
{
    // Given: A 2x3 host vector
    const std::vector<int> shape = { 2, 3 };
    const std::vector<float> host_data = { 1.25F, -2.5F, 3.0F, 4.5F, 0.0F, 6.125F };

    // When: The vector is uploaded to GPU and read back
    Tensor tensor = Tensor::from_host(shape, host_data, Device::GPU);
    synchronize_device();

    // Then: Shape, device, and values are preserved
    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_device(), Device::GPU);
    expect_near_vector(tensor.to_host(), host_data, kTensorEpsilon);
}

TEST_F(GpuTensorTest, FromHostPointerRoundTrip)
{
    // Given: A length-4 C array
    const std::vector<int> shape = { 4 };
    const float host_data[] = { -1.0F, 2.0F, 3.5F, 8.25F };

    // When: The pointer is uploaded to GPU and read back
    Tensor tensor = Tensor::from_host(shape, host_data, Device::GPU);
    synchronize_device();

    // Then: The four values round-trip
    EXPECT_EQ(tensor.get_size(), 4U);
    expect_near_vector(tensor.to_host(), { -1.0F, 2.0F, 3.5F, 8.25F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, FromHostCpuDeviceRoundTrip)
{
    // Given: A 2x2 host buffer destined for CPU
    const std::vector<int> shape = { 2, 2 };
    const std::vector<float> host_data = { 9.0F, 8.0F, 7.0F, 6.0F };

    // When: from_host is called with Device::CPU
    Tensor tensor = Tensor::from_host(shape, host_data, Device::CPU);

    // Then: The tensor stays on CPU and keeps the values
    EXPECT_EQ(tensor.get_device(), Device::CPU);
    expect_near_vector(tensor.to_host(), host_data, kTensorEpsilon);
}

TEST_F(GpuTensorTest, FromHostRejectsMismatchedBufferSize)
{
    // Given: A 3-element buffer for a 2x2 shape
    const std::vector<float> host_data = { 1.0F, 2.0F, 3.0F };

    // When: from_host is called
    // Then: Construction throws
    EXPECT_THROW(Tensor::from_host({ 2, 2 }, host_data, Device::GPU), std::runtime_error);
}

TEST_F(GpuTensorTest, FromHostRejectsNullPointer)
{
    // Given: A null host pointer
    // When: from_host is called
    // Then: Construction throws
    EXPECT_THROW(Tensor::from_host({ 2, 2 }, static_cast<const float*>(nullptr), Device::GPU), std::runtime_error);
}

TEST_F(GpuTensorTest, ZerosLikeMatchesShapeDeviceAndValues)
{
    // Given: A GPU source tensor
    Tensor source = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    // When: zeros_like is created
    Tensor zeros = Tensor::zeros_like(source);
    synchronize_device();

    // Then: Shape and device match and all values are zero
    EXPECT_EQ(zeros.get_shape(), source.get_shape());
    EXPECT_EQ(zeros.get_device(), Device::GPU);
    expect_near_vector(zeros.to_host(), std::vector<float>(6, 0.0F), kTensorEpsilon);
}

TEST_F(GpuTensorTest, SquareMatmul)
{
    // Given: Two 2x2 GPU matrices
    Tensor lhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 5.0F, 6.0F, 7.0F, 8.0F }, Device::GPU);

    // When: The matrices are multiplied
    Tensor result = lhs.matmul(rhs);
    synchronize_device();

    // Then: The product is 2x2 with the expected GEMM values
    EXPECT_EQ(result.get_shape(), (std::vector<int> { 2, 2 }));
    expect_near_vector(result.to_host(), { 19.0F, 22.0F, 43.0F, 50.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, RectangularMatmul)
{
    // Given: A 2x3 matrix and a 3x4 matrix
    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 3, 4 },
        std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
            11.0F, 12.0F },
        Device::GPU);

    // When: The matrices are multiplied
    Tensor result = lhs.matmul(rhs);
    synchronize_device();

    // Then: The product is 2x4 with the expected GEMM values
    EXPECT_EQ(result.get_shape(), (std::vector<int> { 2, 4 }));
    expect_near_vector(result.to_host(), { 38.0F, 44.0F, 50.0F, 56.0F, 83.0F, 98.0F, 113.0F, 128.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, LogicalTransposeMatmulMatchesPhysicalTranspose)
{
    // Given: A 2x3 matrix and a 2x4 matrix that share the batch axis
    Tensor lhs = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 4 },
        { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F }, Device::GPU);

    // When: GEMM uses CUBLAS_OP_T instead of allocating lhs^T
    Tensor logical = lhs.matmul(rhs, true, false);
    Tensor physical = lhs.transpose().matmul(rhs);
    synchronize_device();

    // Then: Shapes and values match the physical transpose path
    EXPECT_EQ(logical.get_shape(), (std::vector<int> { 3, 4 }));
    expect_near_vector(logical.to_host(), physical.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, LogicalTransposeBMatmulMatchesPhysicalTranspose)
{
    // Given: A 2x3 matrix and a 4x3 matrix
    Tensor lhs = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 4, 3 },
        { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F }, Device::GPU);

    // When: GEMM uses CUBLAS_OP_T on B instead of allocating rhs^T
    Tensor logical = lhs.matmul(rhs, false, true);
    Tensor physical = lhs.matmul(rhs.transpose());
    synchronize_device();

    // Then: Shapes and values match the physical transpose path
    EXPECT_EQ(logical.get_shape(), (std::vector<int> { 2, 4 }));
    expect_near_vector(logical.to_host(), physical.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, InPlaceAddMulAndAddScaledDoNotAllocateTemps)
{
    // Given: Two GPU tensors
    Tensor lhs = Tensor::from_host({ 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, { 10.0F, 20.0F, 30.0F, 40.0F }, Device::GPU);
    const float* lhs_ptr = lhs.data();

    // When: In-place add, scale, and scaled-add run
    lhs.add_(rhs);
    lhs.mul_(0.5F);
    lhs.add_scaled_(rhs, 0.1F);
    synchronize_device();

    // Then: The storage pointer is unchanged and values match a + b, * 0.5, + 0.1 b
    EXPECT_EQ(lhs.data(), lhs_ptr);
    expect_near_vector(lhs.to_host(), { 6.5F, 13.0F, 19.5F, 26.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, MatmulIntoAccumulatesIntoExistingBuffer)
{
    // Given: Two 2x2 matrices and a preallocated output filled with ones
    Tensor lhs = Tensor::from_host({ 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, { 5.0F, 6.0F, 7.0F, 8.0F }, Device::GPU);
    Tensor out = Tensor::from_host({ 2, 2 }, { 1.0F, 1.0F, 1.0F, 1.0F }, Device::GPU);
    const float* out_ptr = out.data();

    // When: GEMM writes C = AB + 1 * C
    lhs.matmul_into(rhs, out, false, false, 1.0F);
    synchronize_device();

    // Then: Storage is reused and values are the product plus the previous ones
    EXPECT_EQ(out.data(), out_ptr);
    expect_near_vector(out.to_host(), { 20.0F, 23.0F, 44.0F, 51.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, AddRowBroadcastsBiasAcrossBatch)
{
    // Given: A 2x3 matrix and a 3-wide bias
    Tensor rows = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor bias = Tensor::from_host({ 1, 3 }, { 0.5F, -1.0F, 2.0F }, Device::GPU);

    // When: The bias is added to every row
    rows.add_row_(bias);
    synchronize_device();

    // Then: Each row is shifted by the bias
    expect_near_vector(rows.to_host(), { 1.5F, 1.0F, 5.0F, 4.5F, 4.0F, 8.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, AddSumRowsAccumulatesWithBeta)
{
    // Given: A 2x3 matrix and a [1, 3] accumulator
    Tensor matrix = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor acc = Tensor::from_host({ 1, 3 }, { 10.0F, 10.0F, 10.0F }, Device::GPU);

    // When: Columns are summed into acc with beta = 0.5
    acc.add_sum_rows_(matrix, 0.5F);
    synchronize_device();

    // Then: acc[j] = 0.5 * 10 + sum of column j
    expect_near_vector(acc.to_host(), { 10.0F, 12.0F, 14.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, IdentityMatmulLeavesMatrixUnchanged)
{
    // Given: A 3x3 matrix and a 3x3 identity
    Tensor matrix = Tensor::from_host({ 3, 3 },
        std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F },
        Device::GPU);
    Tensor identity = Tensor::from_host({ 3, 3 },
        std::vector<float> { 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F },
        Device::GPU);

    // When: The matrix is multiplied by identity
    Tensor result = matrix.matmul(identity);
    synchronize_device();

    // Then: The values are unchanged
    expect_near_vector(result.to_host(), matrix.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, MatmulMismatchedInnerDimensionsThrows)
{
    // Given: Matrices whose inner dimensions do not match
    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 4, 5 }, std::vector<float>(20, 1.0F), Device::GPU);

    // When: matmul is called
    // Then: The operation throws
    EXPECT_THROW(lhs.matmul(rhs), std::runtime_error);
}

TEST_F(GpuTensorTest, ElementwiseTensorAddition)
{
    // Given: Two 2x2 GPU tensors
    Tensor lhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 10.0F, 20.0F, 30.0F, 40.0F }, Device::GPU);

    // When: They are added
    Tensor result = lhs + rhs;
    synchronize_device();

    // Then: Each element is the pairwise sum
    expect_near_vector(result.to_host(), { 11.0F, 22.0F, 33.0F, 44.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ElementwiseTensorSubtraction)
{
    // Given: Two length-4 GPU tensors
    Tensor lhs = Tensor::from_host({ 4 }, std::vector<float> { 5.0F, 4.0F, 3.0F, 2.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 4 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 8.0F }, Device::GPU);

    // When: They are subtracted
    Tensor result = lhs - rhs;
    synchronize_device();

    // Then: Each element is the pairwise difference
    expect_near_vector(result.to_host(), { 4.0F, 2.0F, 0.0F, -6.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ElementwiseTensorMultiplication)
{
    // Given: Two 2x3 GPU tensors
    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, -2.0F, 3.0F, 4.0F, 0.5F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 2.0F, 3.0F, 4.0F, 0.0F, 8.0F, -1.0F }, Device::GPU);

    // When: They are multiplied elementwise
    Tensor result = lhs * rhs;
    synchronize_device();

    // Then: Each element is the pairwise product
    expect_near_vector(result.to_host(), { 2.0F, -6.0F, 12.0F, 0.0F, 4.0F, -6.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, TensorScalarMultiplicationAndAddition)
{
    // Given: A length-3 GPU tensor
    Tensor tensor = Tensor::from_host({ 3 }, std::vector<float> { 1.0F, -2.0F, 4.0F }, Device::GPU);

    // When: The tensor is scaled by 2.5 and shifted by 1
    Tensor scaled = tensor * 2.5F;
    Tensor shifted = tensor + 1.0F;
    synchronize_device();

    // Then: Broadcast scalar ops match the expected values
    expect_near_vector(scaled.to_host(), { 2.5F, -5.0F, 10.0F }, kTensorEpsilon);
    expect_near_vector(shifted.to_host(), { 2.0F, -1.0F, 5.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ElementwiseSizeMismatchThrows)
{
    // Given: Tensors with different element counts
    Tensor lhs = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: Elementwise ops are invoked
    // Then: Each op throws
    EXPECT_THROW(lhs + rhs, std::runtime_error);
    EXPECT_THROW(lhs - rhs, std::runtime_error);
    EXPECT_THROW(lhs * rhs, std::runtime_error);
}

TEST_F(GpuTensorTest, ElementwiseCpuOperandThrows)
{
    // Given: A GPU tensor and a CPU tensor of the same shape
    Tensor gpu = Tensor::from_host({ 2 }, std::vector<float> { 1.0F, 2.0F }, Device::GPU);
    Tensor cpu = Tensor::from_host({ 2 }, std::vector<float> { 3.0F, 4.0F }, Device::CPU);

    // When: Mixed-device or CPU-only elementwise ops are invoked
    // Then: The ops throw
    EXPECT_THROW(gpu + cpu, std::runtime_error);
    EXPECT_THROW(cpu * 2.0F, std::runtime_error);
}

TEST_F(GpuTensorTest, ClampBoundsValues)
{
    // Given: A tensor with values outside [-1, 1]
    Tensor tensor = Tensor::from_host({ 5 }, std::vector<float> { -4.0F, -0.5F, 0.0F, 0.7F, 3.0F }, Device::GPU);

    // When: clamp(-1, 1) is applied
    Tensor clamped = tensor.clamp(-1.0F, 1.0F);
    synchronize_device();

    // Then: Values are clipped to the closed interval
    expect_near_vector(clamped.to_host(), { -1.0F, -0.5F, 0.0F, 0.7F, 1.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ClampInvalidRangeThrows)
{
    // Given: A GPU tensor and an inverted clamp range
    Tensor tensor = Tensor::from_host({ 2 }, std::vector<float> { 1.0F, 2.0F }, Device::GPU);

    // When: clamp is called with min > max
    // Then: The operation throws
    EXPECT_THROW(tensor.clamp(1.0F, 0.0F), std::runtime_error);
}

TEST_F(GpuTensorTest, GlobalSumPositiveValues)
{
    // Given: A 2x2 tensor of positive values
    Tensor tensor = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: A global sum is computed
    Tensor reduced = tensor.sum();
    synchronize_device();

    // Then: The result is a scalar 10
    EXPECT_EQ(reduced.get_shape(), (std::vector<int> { 1 }));
    expect_near_vector(reduced.to_host(), { 10.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, GlobalSumNegativeAndMixedValues)
{
    // Given: A mixed-sign vector
    Tensor tensor = Tensor::from_host({ 4 }, std::vector<float> { 1.0F, -2.0F, 3.0F, -4.5F }, Device::GPU);

    // When: sum(-1) reduces all elements
    Tensor reduced = tensor.sum(-1);
    synchronize_device();

    // Then: The total is -2.5
    expect_near_vector(reduced.to_host(), { -2.5F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, SumAlongAxisThrows)
{
    // Given: A 1D tensor
    Tensor tensor = Tensor::from_host({ 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F }, Device::GPU);

    // When: An axis-wise sum is requested
    // Then: The unsupported axis reduction throws
    EXPECT_THROW(tensor.sum(0), std::runtime_error);
}

TEST_F(GpuTensorTest, ViewReshapeSharesStorageAndValues)
{
    // Given: A contiguous 2x3 GPU tensor
    Tensor tensor = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    // When: It is viewed as 3x2
    Tensor viewed = tensor.view({ 3, 2 });

    // Then: Storage is shared and values are unchanged
    EXPECT_EQ(viewed.get_shape(), (std::vector<int> { 3, 2 }));
    EXPECT_EQ(viewed.get_strides(), (std::vector<int> { 2, 1 }));
    EXPECT_EQ(viewed.get_size(), tensor.get_size());
    EXPECT_EQ(viewed.data(), tensor.data());
    expect_near_vector(viewed.to_host(), tensor.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, ViewInfersMinusOneDimension)
{
    // Given: A 2x4 tensor
    Tensor tensor = Tensor::from_host({ 2, 4 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F },
        Device::GPU);

    // When: view({-1, 2}) infers the leading dimension
    Tensor viewed = tensor.view({ -1, 2 });

    // Then: The inferred shape is 4x2
    EXPECT_EQ(viewed.get_shape(), (std::vector<int> { 4, 2 }));
    expect_near_vector(viewed.to_host(), tensor.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, ViewIncompatibleShapeThrows)
{
    // Given: A 2x3 tensor
    Tensor tensor = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    // When: An incompatible or ambiguous view is requested
    // Then: The view throws
    EXPECT_THROW(tensor.view({ 2, 2 }), std::runtime_error);
    EXPECT_THROW(tensor.view({ -1, -1 }), std::runtime_error);
}

TEST_F(GpuTensorTest, TransposeTwoDimensional)
{
    // Given: A 2x3 GPU tensor
    Tensor tensor = Tensor::from_host({ 2, 3 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    // When: The tensor is transposed
    Tensor transposed = tensor.transpose();
    synchronize_device();

    // Then: The shape is 3x2 with column-major values materialized
    EXPECT_EQ(transposed.get_shape(), (std::vector<int> { 3, 2 }));
    expect_near_vector(transposed.to_host(), { 1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, TransposeThenMatmulMatchesOriginalProductLayout)
{
    // Given: A 2x2 GPU matrix
    Tensor matrix = Tensor::from_host({ 2, 2 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: The matrix is transposed
    Tensor transposed = matrix.transpose();
    synchronize_device();

    // Then: The layout is the 2x2 transpose
    expect_near_vector(transposed.to_host(), { 1.0F, 3.0F, 2.0F, 4.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, TransposeRejectsNonTwoDimensionalTensors)
{
    // Given: A vector and a 3D tensor
    Tensor vector = Tensor::from_host({ 4 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor volume = Tensor::from_host({ 2, 2, 1 }, std::vector<float> { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: transpose is called
    // Then: Both ranks are rejected
    EXPECT_THROW(vector.transpose(), std::runtime_error);
    EXPECT_THROW(volume.transpose(), std::runtime_error);
}

TEST_F(GpuTensorTest, VectorMatrixMatmul)
{
    // Given: A length-3 vector and a 3x2 matrix
    Tensor vector = Tensor::from_host({ 3 }, { 1.0F, 2.0F, 3.0F }, Device::GPU);
    Tensor matrix = Tensor::from_host({ 3, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F }, Device::GPU);

    // When: The vector is multiplied by the matrix
    Tensor result = vector.matmul(matrix);
    synchronize_device();

    // Then: The result is [4, 5]
    EXPECT_EQ(result.get_shape(), (std::vector<int> { 2 }));
    expect_near_vector(result.to_host(), { 4.0F, 5.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, MatmulOnCpuThrows)
{
    // Given: Two CPU matrices
    Tensor lhs = Tensor::from_host({ 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F }, Device::CPU);
    Tensor rhs = Tensor::from_host({ 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::CPU);

    // When: matmul is called
    // Then: The GPU-only op throws
    EXPECT_THROW(lhs.matmul(rhs), std::runtime_error);
}

TEST_F(GpuTensorTest, ChainedElementwiseOps)
{
    // Given: Three equal-sized GPU tensors
    Tensor a = Tensor::from_host({ 3 }, { 1.0F, 2.0F, 3.0F }, Device::GPU);
    Tensor b = Tensor::from_host({ 3 }, { 4.0F, 5.0F, 6.0F }, Device::GPU);
    Tensor c = Tensor::from_host({ 3 }, { 2.0F, 2.0F, 2.0F }, Device::GPU);

    // When: (a + b) * c is evaluated
    Tensor result = (a + b) * c;
    synchronize_device();

    // Then: Each element is 2 * (a + b)
    expect_near_vector(result.to_host(), { 10.0F, 14.0F, 18.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ClampEqualBoundsSaturates)
{
    // Given: A tensor and a degenerate [2, 2] clamp range
    Tensor tensor = Tensor::from_host({ 3 }, { 0.0F, 2.0F, 5.0F }, Device::GPU);

    // When: clamp(2, 2) is applied
    Tensor clamped = tensor.clamp(2.0F, 2.0F);
    synchronize_device();

    // Then: Every value becomes 2
    expect_near_vector(clamped.to_host(), { 2.0F, 2.0F, 2.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ViewFlattensWithMinusOne)
{
    // Given: A 2x2x2 tensor
    Tensor tensor = Tensor::from_host({ 2, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F }, Device::GPU);

    // When: It is viewed as a vector
    Tensor viewed = tensor.view({ -1 });

    // Then: The shape is length 8
    EXPECT_EQ(viewed.get_shape(), (std::vector<int> { 8 }));
    expect_near_vector(viewed.to_host(), tensor.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, ViewRejectsInvalidNegativeDimension)
{
    // Given: A length-4 tensor
    Tensor tensor = Tensor::from_host({ 4 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: view uses a negative axis other than -1
    // Then: The view throws
    EXPECT_THROW(tensor.view({ -2 }), std::runtime_error);
}

TEST_F(GpuTensorTest, ViewIdentityKeepsShape)
{
    // Given: A 2x3 tensor
    Tensor tensor = Tensor::from_host({ 2, 3 }, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Device::GPU);

    // When: It is viewed as the same shape
    Tensor viewed = tensor.view({ 2, 3 });

    // Then: Pointer, shape, and values are unchanged
    EXPECT_EQ(viewed.data(), tensor.data());
    EXPECT_EQ(viewed.get_shape(), tensor.get_shape());
    expect_near_vector(viewed.to_host(), tensor.to_host(), kTensorEpsilon);
}

TEST_F(GpuTensorTest, SumOfZerosIsZero)
{
    // Given: A zero-filled GPU tensor
    Tensor tensor = Tensor::from_host({ 3, 3 }, std::vector<float>(9, 0.0F), Device::GPU);

    // When: A global sum is computed
    Tensor reduced = tensor.sum();
    synchronize_device();

    // Then: The scalar is 0
    expect_near_vector(reduced.to_host(), { 0.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, TransposeOneByN)
{
    // Given: A 1x4 row vector
    Tensor row = Tensor::from_host({ 1, 4 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: It is transposed
    Tensor column = row.transpose();
    synchronize_device();

    // Then: The shape is 4x1 with the same values
    EXPECT_EQ(column.get_shape(), (std::vector<int> { 4, 1 }));
    expect_near_vector(column.to_host(), { 1.0F, 2.0F, 3.0F, 4.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, MoveConstructorTransfersOwnership)
{
    // Given: A GPU tensor with known values
    Tensor original = Tensor::from_host({ 2 }, { 7.0F, 8.0F }, Device::GPU);
    const float* pointer = original.data();

    // When: The tensor is moved
    Tensor moved(std::move(original));
    synchronize_device();

    // Then: The destination owns the same storage and values
    EXPECT_EQ(moved.data(), pointer);
    expect_near_vector(moved.to_host(), { 7.0F, 8.0F }, kTensorEpsilon);
}

TEST_F(GpuTensorTest, ZerosLikeCpuSourceStaysOnCpu)
{
    // Given: A CPU source tensor
    Tensor source = Tensor::from_host({ 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::CPU);

    // When: zeros_like is created
    Tensor zeros = Tensor::zeros_like(source);

    // Then: The result is a CPU zero tensor of the same shape
    EXPECT_EQ(zeros.get_device(), Device::CPU);
    EXPECT_EQ(zeros.get_shape(), source.get_shape());
    expect_near_vector(zeros.to_host(), std::vector<float>(4, 0.0F), kTensorEpsilon);
}

TEST_F(GpuTensorTest, CpuToHostRoundTrip)
{
    // Given: A CPU tensor
    Tensor tensor = Tensor::from_host({ 3 }, { 1.5F, 2.5F, 3.5F }, Device::CPU);

    // When: to_host is called
    const std::vector<float> host = tensor.to_host();

    // Then: Values match without a device copy changing them
    expect_near_vector(host, { 1.5F, 2.5F, 3.5F }, kTensorEpsilon);
}
