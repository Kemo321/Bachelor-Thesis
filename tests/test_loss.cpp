#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace dl;

namespace
{
constexpr float kEpsilon = 1e-5F;
constexpr int kGrid = 7;
constexpr int kNumClasses = 20;
constexpr int kAttributes = 10 + kNumClasses;

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

auto grid_index(int row, int col, int offset) -> size_t
{
    return static_cast<size_t>((row * kGrid * kAttributes) + (col * kAttributes) + offset);
}

auto zeros_grid(int batch) -> std::vector<float>
{
    return std::vector<float>(static_cast<size_t>(batch * kGrid * kGrid * kAttributes), 0.0F);
}

auto expect_all_finite(const std::vector<float>& values) -> void
{
    for (size_t index = 0; index < values.size(); ++index)
    {
        EXPECT_TRUE(std::isfinite(values[index])) << "non-finite value at index " << index;
    }
}
} // namespace

class GpuLossTest : public ::testing::Test
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

TEST_F(GpuLossTest, LossOnEmptyGridIsNearZeroScalar)
{
    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);

    Tensor loss = YOLOLoss::loss(target, prediction, kNumClasses);
    synchronize_device();

    EXPECT_EQ(loss.get_shape(), (std::vector<int> { 1 }));
    EXPECT_EQ(loss.get_device(), Device::GPU);
    const std::vector<float> host = loss.to_host();
    ASSERT_EQ(host.size(), 1U);
    EXPECT_TRUE(std::isfinite(host[0]));
    EXPECT_NEAR(host[0], 0.0F, 1e-4F);
}

TEST_F(GpuLossTest, LossIsPositiveWhenPredictionMissesAnObject)
{
    std::vector<float> target_host = zeros_grid(1);
    target_host[grid_index(3, 3, 0)] = 0.5F;
    target_host[grid_index(3, 3, 1)] = 0.5F;
    target_host[grid_index(3, 3, 2)] = 0.2F;
    target_host[grid_index(3, 3, 3)] = 0.2F;
    target_host[grid_index(3, 3, 4)] = 1.0F;
    target_host[grid_index(3, 3, 10)] = 1.0F;

    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, target_host, Device::GPU);

    Tensor loss = YOLOLoss::loss(target, prediction, kNumClasses);
    synchronize_device();

    const std::vector<float> host = loss.to_host();
    ASSERT_EQ(host.size(), 1U);
    EXPECT_TRUE(std::isfinite(host[0]));
    EXPECT_GT(host[0], 0.0F);
}

TEST_F(GpuLossTest, MatchingObjectPredictionHasSmallerLossThanZeros)
{
    std::vector<float> target_host = zeros_grid(1);
    target_host[grid_index(2, 4, 0)] = 0.4F;
    target_host[grid_index(2, 4, 1)] = 0.6F;
    target_host[grid_index(2, 4, 2)] = 0.3F;
    target_host[grid_index(2, 4, 3)] = 0.25F;
    target_host[grid_index(2, 4, 4)] = 1.0F;
    target_host[grid_index(2, 4, 10)] = 1.0F;

    std::vector<float> pred_host = target_host;
    pred_host[grid_index(2, 4, 4)] = 1.0F;

    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, target_host, Device::GPU);
    Tensor matching = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, pred_host, Device::GPU);
    Tensor zeros = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);

    const float match_loss = YOLOLoss::loss(target, matching, kNumClasses).to_host().front();
    const float zero_loss = YOLOLoss::loss(target, zeros, kNumClasses).to_host().front();
    synchronize_device();

    EXPECT_TRUE(std::isfinite(match_loss));
    EXPECT_TRUE(std::isfinite(zero_loss));
    EXPECT_LT(match_loss, zero_loss);
}

TEST_F(GpuLossTest, LossDerivativeMatchesPredictionShapeAndIsFinite)
{
    std::vector<float> pred_host = zeros_grid(1);
    std::vector<float> target_host = zeros_grid(1);
    target_host[grid_index(0, 1, 0)] = 0.2F;
    target_host[grid_index(0, 1, 1)] = 0.3F;
    target_host[grid_index(0, 1, 2)] = 0.1F;
    target_host[grid_index(0, 1, 3)] = 0.1F;
    target_host[grid_index(0, 1, 4)] = 1.0F;
    target_host[grid_index(0, 1, 11)] = 1.0F;
    pred_host[grid_index(0, 1, 0)] = 0.25F;
    pred_host[grid_index(0, 1, 1)] = 0.28F;
    pred_host[grid_index(0, 1, 2)] = 0.12F;
    pred_host[grid_index(0, 1, 3)] = 0.09F;
    pred_host[grid_index(0, 1, 4)] = 0.8F;
    pred_host[grid_index(0, 1, 11)] = 0.7F;

    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, pred_host, Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, target_host, Device::GPU);

    Tensor gradient = YOLOLoss::loss_derivative(target, prediction, kNumClasses);
    synchronize_device();

    EXPECT_EQ(gradient.get_shape(), prediction.get_shape());
    EXPECT_EQ(gradient.get_device(), Device::GPU);
    const std::vector<float> grad_host = gradient.to_host();
    EXPECT_EQ(grad_host.size(), pred_host.size());
    expect_all_finite(grad_host);

    bool any_nonzero = false;
    for (float value : grad_host)
    {
        if (std::fabs(value) > kEpsilon)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

TEST_F(GpuLossTest, FlattenedLayoutIsAcceptedAndGradientsMatchRank)
{
    std::vector<float> host = zeros_grid(2);
    host[grid_index(1, 1, 4)] = 1.0F;
    host[grid_index(1, 1, 10)] = 1.0F;

    Tensor prediction = Tensor::from_host({ 2, kGrid * kGrid * kAttributes }, host, Device::GPU);
    Tensor target = Tensor::from_host({ 2, kGrid * kGrid * kAttributes }, host, Device::GPU);

    Tensor loss = YOLOLoss::loss(target, prediction, kNumClasses);
    Tensor gradient = YOLOLoss::loss_derivative(target, prediction, kNumClasses);
    synchronize_device();

    EXPECT_EQ(loss.get_shape(), (std::vector<int> { 1 }));
    EXPECT_TRUE(std::isfinite(loss.to_host().front()));
    EXPECT_EQ(gradient.get_shape(), prediction.get_shape());
    expect_all_finite(gradient.to_host());
}

TEST_F(GpuLossTest, BatchSizeMismatchThrows)
{
    Tensor prediction = Tensor::from_host({ 2, kGrid, kGrid, kAttributes }, zeros_grid(2), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);

    EXPECT_THROW(YOLOLoss::loss(target, prediction, kNumClasses), std::runtime_error);
    EXPECT_THROW(YOLOLoss::loss_derivative(target, prediction, kNumClasses), std::runtime_error);
}

TEST_F(GpuLossTest, InvalidClassCountThrows)
{
    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);

    EXPECT_THROW(YOLOLoss::loss(target, prediction, 0), std::runtime_error);
    EXPECT_THROW(YOLOLoss::loss_derivative(target, prediction, -1), std::runtime_error);
}
