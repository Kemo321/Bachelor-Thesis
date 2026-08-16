#include "test_helpers.hpp"

#include "DeepLearnLib/Profiler.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/mAP.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace dl;
using namespace dllib_test;

TEST(MapTest, PerfectOverlapYieldsUnitMap)
{
    // Given: Identical predicted and ground-truth boxes for one class
    const Detection box { 10.0F, 20.0F, 30.0F, 40.0F, 0.9F, 1 };
    const std::vector<Detection> predicted = { box };
    const std::vector<Detection> ground_truth = { box };

    // When: mAP@0.5 is computed
    const float map = mean_average_precision(predicted, ground_truth, 0.5F);

    // Then: Average precision is 1
    EXPECT_NEAR(map, 1.0F, kEpsilon);
}

TEST(MapTest, DisjointBoxesYieldZeroMap)
{
    // Given: A prediction that does not overlap the ground truth
    const Detection predicted_box { 0.0F, 0.0F, 10.0F, 10.0F, 0.8F, 0 };
    const Detection ground_truth_box { 50.0F, 50.0F, 10.0F, 10.0F, 1.0F, 0 };

    // When: mAP@0.5 is computed
    const float map = mean_average_precision({ predicted_box }, { ground_truth_box }, 0.5F);

    // Then: There are no true positives
    EXPECT_NEAR(map, 0.0F, kEpsilon);
}

TEST(MapTest, EmptyGroundTruthIsZeroAndInvalidThresholdThrows)
{
    // Given: Predictions but no ground truth
    const Detection predicted_box { 0.0F, 0.0F, 5.0F, 5.0F, 0.4F, 2 };

    // When: mAP is requested without labels
    const float map = mean_average_precision({ predicted_box }, {}, 0.5F);

    // Then: The metric is defined as zero and a bad threshold is rejected
    EXPECT_NEAR(map, 0.0F, kEpsilon);
    EXPECT_THROW(static_cast<void>(mean_average_precision({ predicted_box }, { predicted_box }, 1.5F)),
        std::runtime_error);
}

TEST(MapTest, DetectionIouIsOneForIdenticalBoxes)
{
    // Given: Two copies of the same box
    const Detection box { 1.0F, 2.0F, 3.0F, 4.0F, 0.5F, 0 };

    // When: IoU is computed
    const float iou = detection_iou(box, box);

    // Then: The boxes fully overlap
    EXPECT_NEAR(iou, 1.0F, kEpsilon);
}

class ProfilerTest : public GpuTest
{
};

TEST_F(ProfilerTest, StopReturnsNonNegativeGpuMilliseconds)
{
    // Given: A profiler and a GPU reduction that will be timed
    Profiler profiler;
    Tensor values = Tensor::from_host({ 1024 }, std::vector<float>(1024, 1.0F), Device::GPU);

    // When: The reduction is bracketed by CUDA events
    profiler.start();
    Tensor total = values.sum();
    synchronize_device();
    const float elapsed_ms = profiler.stop();

    // Then: Elapsed time is finite and non-negative, and the reduction is correct
    EXPECT_GE(elapsed_ms, 0.0F);
    EXPECT_TRUE(std::isfinite(elapsed_ms));
    expect_near_vector(total.to_host(), { 1024.0F }, kLooseEpsilon);
}

TEST_F(ProfilerTest, VramUsageIsReportedInMebibytes)
{
    // Given: A live CUDA context after a GPU allocation
    Tensor buffer({ 256, 256, 8 }, Device::GPU);
    synchronize_device();

    // When: Process VRAM is queried
    const std::size_t used_mb = Profiler::get_vram_usage_mb();

    // Then: The device reports a non-zero allocation footprint
    EXPECT_GT(used_mb, 0U);
}

TEST_F(ProfilerTest, StopWithoutStartThrows)
{
    // Given: A profiler that has not been started
    Profiler profiler;

    // When / Then: stop() is rejected
    EXPECT_THROW(static_cast<void>(profiler.stop()), std::runtime_error);
}
