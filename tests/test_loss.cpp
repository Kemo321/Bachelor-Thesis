#include "test_helpers.hpp"

#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace dl;
using namespace dllib_test;

namespace
{
constexpr int kGrid = 7;
constexpr int kNumClasses = 20;
constexpr int kAttributes = 10 + kNumClasses;

auto grid_index(int row, int col, int offset) -> size_t
{
    return static_cast<size_t>((row * kGrid * kAttributes) + (col * kAttributes) + offset);
}

auto zeros_grid(int batch) -> std::vector<float>
{
    return std::vector<float>(static_cast<size_t>(batch * kGrid * kGrid * kAttributes), 0.0F);
}
} // namespace

class GpuLossTest : public GpuTest
{
};

TEST_F(GpuLossTest, LossOnEmptyGridIsNearZeroScalar)
{
    // Given: An empty YOLOv1 prediction and target grid
    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);

    // When: The loss is computed
    Tensor loss = YOLOLoss::loss(target, prediction, kNumClasses);
    synchronize_device();

    // Then: The result is a finite GPU scalar near zero
    EXPECT_EQ(loss.get_shape(), (std::vector<int> { 1 }));
    EXPECT_EQ(loss.get_device(), Device::GPU);
    const std::vector<float> host = loss.to_host();
    ASSERT_EQ(host.size(), 1U);
    EXPECT_TRUE(std::isfinite(host[0]));
    EXPECT_NEAR(host[0], 0.0F, 1e-4F);
}

TEST_F(GpuLossTest, LossIsPositiveWhenPredictionMissesAnObject)
{
    // Given: A target object in cell (3, 3) and an all-zero prediction
    std::vector<float> target_host = zeros_grid(1);
    target_host[grid_index(3, 3, 0)] = 0.5F;
    target_host[grid_index(3, 3, 1)] = 0.5F;
    target_host[grid_index(3, 3, 2)] = 0.2F;
    target_host[grid_index(3, 3, 3)] = 0.2F;
    target_host[grid_index(3, 3, 4)] = 1.0F;
    target_host[grid_index(3, 3, 10)] = 1.0F;
    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, target_host, Device::GPU);

    // When: The loss is computed
    Tensor loss = YOLOLoss::loss(target, prediction, kNumClasses);
    synchronize_device();

    // Then: The scalar loss is finite and strictly positive
    const std::vector<float> host = loss.to_host();
    ASSERT_EQ(host.size(), 1U);
    EXPECT_TRUE(std::isfinite(host[0]));
    EXPECT_GT(host[0], 0.0F);
}

TEST_F(GpuLossTest, MatchingObjectPredictionHasSmallerLossThanZeros)
{
    // Given: A target object and both a matching prediction and a zero prediction
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

    // When: Loss is evaluated for both predictions
    const float match_loss = YOLOLoss::loss(target, matching, kNumClasses).to_host().front();
    const float zero_loss = YOLOLoss::loss(target, zeros, kNumClasses).to_host().front();
    synchronize_device();

    // Then: The matching prediction has a smaller finite loss
    EXPECT_TRUE(std::isfinite(match_loss));
    EXPECT_TRUE(std::isfinite(zero_loss));
    EXPECT_LT(match_loss, zero_loss);
}

TEST_F(GpuLossTest, LossDerivativeMatchesPredictionShapeAndIsFinite)
{
    // Given: A slightly mismatched object prediction and target
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

    // When: The loss derivative is computed
    Tensor gradient = YOLOLoss::loss_derivative(target, prediction, kNumClasses);
    synchronize_device();

    // Then: The gradient matches the prediction layout, is finite, and is not all zeros
    EXPECT_EQ(gradient.get_shape(), prediction.get_shape());
    EXPECT_EQ(gradient.get_device(), Device::GPU);
    const std::vector<float> grad_host = gradient.to_host();
    EXPECT_EQ(grad_host.size(), pred_host.size());
    expect_all_finite(grad_host);

    bool any_nonzero = false;
    for (float value : grad_host)
    {
        if (std::fabs(value) > kTensorEpsilon)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

TEST_F(GpuLossTest, FlattenedLayoutIsAcceptedAndGradientsMatchRank)
{
    // Given: Flattened [N, S*S*(B*5+C)] prediction and target tensors
    std::vector<float> host = zeros_grid(2);
    host[grid_index(1, 1, 4)] = 1.0F;
    host[grid_index(1, 1, 10)] = 1.0F;
    Tensor prediction = Tensor::from_host({ 2, kGrid * kGrid * kAttributes }, host, Device::GPU);
    Tensor target = Tensor::from_host({ 2, kGrid * kGrid * kAttributes }, host, Device::GPU);

    // When: Loss and derivative are computed
    Tensor loss = YOLOLoss::loss(target, prediction, kNumClasses);
    Tensor gradient = YOLOLoss::loss_derivative(target, prediction, kNumClasses);
    synchronize_device();

    // Then: Loss is a finite scalar and the gradient keeps the flattened rank
    EXPECT_EQ(loss.get_shape(), (std::vector<int> { 1 }));
    EXPECT_TRUE(std::isfinite(loss.to_host().front()));
    EXPECT_EQ(gradient.get_shape(), prediction.get_shape());
    expect_all_finite(gradient.to_host());
}

TEST_F(GpuLossTest, BatchSizeMismatchThrows)
{
    // Given: Prediction and target tensors with different batch sizes
    Tensor prediction = Tensor::from_host({ 2, kGrid, kGrid, kAttributes }, zeros_grid(2), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);

    // When: Loss or derivative is computed
    // Then: Both entry points throw
    EXPECT_THROW(YOLOLoss::loss(target, prediction, kNumClasses), std::runtime_error);
    EXPECT_THROW(YOLOLoss::loss_derivative(target, prediction, kNumClasses), std::runtime_error);
}

TEST_F(GpuLossTest, InvalidClassCountThrows)
{
    // Given: Valid tensors and a non-positive class count
    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);

    // When: Loss or derivative is computed with an invalid class count
    // Then: Both entry points throw
    EXPECT_THROW(YOLOLoss::loss(target, prediction, 0), std::runtime_error);
    EXPECT_THROW(YOLOLoss::loss_derivative(target, prediction, -1), std::runtime_error);
}

TEST_F(GpuLossTest, CpuTensorsThrow)
{
    // Given: CPU prediction and target grids
    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::CPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::CPU);

    // When: Loss is computed
    // Then: GPU-only validation throws
    EXPECT_THROW(YOLOLoss::loss(target, prediction, kNumClasses), std::runtime_error);
}

TEST_F(GpuLossTest, RankThreeAndWrongGridThrow)
{
    // Given: Illegal YOLO layouts
    Tensor rank3 = Tensor::from_host({ 1, kGrid, kAttributes }, std::vector<float>(kGrid * kAttributes, 0.0F),
        Device::GPU);
    Tensor bad_grid = Tensor::from_host({ 1, 3, 3, kAttributes },
        std::vector<float>(static_cast<size_t>(3 * 3 * kAttributes), 0.0F), Device::GPU);
    Tensor bad_flat = Tensor::from_host({ 1, 10 }, std::vector<float>(10, 0.0F), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);

    // When: Loss is computed
    // Then: Layout validation throws
    EXPECT_THROW(YOLOLoss::loss(target, rank3, kNumClasses), std::runtime_error);
    EXPECT_THROW(YOLOLoss::loss(target, bad_grid, kNumClasses), std::runtime_error);
    EXPECT_THROW(YOLOLoss::loss(target, bad_flat, kNumClasses), std::runtime_error);
}

TEST_F(GpuLossTest, SingleClassGridIsAccepted)
{
    // Given: A 1-class empty grid
    constexpr int one_class = 1;
    constexpr int attrs = 10 + one_class;
    std::vector<float> zeros(static_cast<size_t>(kGrid * kGrid * attrs), 0.0F);
    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, attrs }, zeros, Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, attrs }, zeros, Device::GPU);

    // When: Loss and derivative are computed
    Tensor loss = YOLOLoss::loss(target, prediction, one_class);
    Tensor grad = YOLOLoss::loss_derivative(target, prediction, one_class);
    synchronize_device();

    // Then: Loss is a near-zero scalar and the gradient matches the grid
    EXPECT_NEAR(loss.to_host().front(), 0.0F, 1e-4F);
    EXPECT_EQ(grad.get_shape(), prediction.get_shape());
}

TEST_F(GpuLossTest, TwoObjectsYieldFinitePositiveLoss)
{
    // Given: Two occupied cells and a zero prediction
    std::vector<float> target_host = zeros_grid(1);
    target_host[grid_index(1, 1, 4)] = 1.0F;
    target_host[grid_index(1, 1, 10)] = 1.0F;
    target_host[grid_index(5, 2, 0)] = 0.3F;
    target_host[grid_index(5, 2, 1)] = 0.4F;
    target_host[grid_index(5, 2, 2)] = 0.2F;
    target_host[grid_index(5, 2, 3)] = 0.2F;
    target_host[grid_index(5, 2, 4)] = 1.0F;
    target_host[grid_index(5, 2, 11)] = 1.0F;
    Tensor prediction = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, zeros_grid(1), Device::GPU);
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, target_host, Device::GPU);

    // When: Loss is computed
    const float loss = YOLOLoss::loss(target, prediction, kNumClasses).to_host().front();
    synchronize_device();

    // Then: The scalar is finite and larger than a single-object miss
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GT(loss, 0.0F);
}

TEST_F(GpuLossTest, MixedRankLayoutsAreAcceptedTogether)
{
    // Given: A rank-4 target and a flattened prediction of the same batch
    std::vector<float> host = zeros_grid(1);
    host[grid_index(0, 0, 4)] = 1.0F;
    host[grid_index(0, 0, 10)] = 1.0F;
    Tensor target = Tensor::from_host({ 1, kGrid, kGrid, kAttributes }, host, Device::GPU);
    Tensor prediction = Tensor::from_host({ 1, kGrid * kGrid * kAttributes }, host, Device::GPU);

    // When: Loss is computed
    Tensor loss = YOLOLoss::loss(target, prediction, kNumClasses);
    synchronize_device();

    // Then: The mixed layouts produce a finite scalar
    EXPECT_TRUE(std::isfinite(loss.to_host().front()));
}
