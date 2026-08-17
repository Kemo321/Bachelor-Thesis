#include "test_helpers.hpp"

#include "DeepLearnLib/ClassificationLoader.hpp"
#include "DeepLearnLib/ParallelFor.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

namespace
{

auto unique_temp_root() -> std::filesystem::path
{
    static std::atomic<int> counter { 0 };
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path()
        / ("dllib_clf_" + std::to_string(stamp) + "_" + std::to_string(counter.fetch_add(1)));
}

auto write_solid_png(const std::filesystem::path& path, unsigned char red, unsigned char green, unsigned char blue)
    -> void
{
    cv::Mat image(8, 8, CV_8UC3);
    image.setTo(cv::Scalar(blue, green, red));
    std::filesystem::create_directories(path.parent_path());
    if (!cv::imwrite(path.string(), image))
    {
        throw std::runtime_error("cv::imwrite failed: " + path.string());
    }
}

} // namespace

class ClassificationLoaderTest : public GpuTest
{
protected:
    std::filesystem::path root_;

    void SetUp() override
    {
        GpuTest::SetUp();
        if (IsSkipped())
        {
            return;
        }
        root_ = unique_temp_root();
        write_solid_png(root_ / "train" / "cat" / "a.png", 255, 0, 0);
        write_solid_png(root_ / "train" / "cat" / "b.png", 255, 0, 0);
        write_solid_png(root_ / "train" / "dog" / "c.png", 0, 0, 255);
        write_solid_png(root_ / "train" / "dog" / "d.png", 0, 0, 255);
        write_solid_png(root_ / "train" / "dog" / "e.png", 0, 0, 255);
    }

    void TearDown() override
    {
        if (!root_.empty())
        {
            std::error_code error;
            std::filesystem::remove_all(root_, error);
        }
    }
};

TEST_F(ClassificationLoaderTest, DiscoversClassesAndYieldsNchwOneHotBatches)
{
    // Given: Five PNG images in two class folders
    ClassificationLoader loader((root_ / "train").string(), "train", 2, 8, false);

    // When: The first batch is requested
    EXPECT_EQ(loader.size(), 5U);
    EXPECT_EQ(loader.num_classes(), 2);
    EXPECT_EQ(loader.class_names(), (std::vector<std::string> { "cat", "dog" }));
    Batch batch = loader.get_batch();
    synchronize_device();

    // Then: Images are NCHW on the GPU and targets are one-hot rows
    EXPECT_EQ(batch.images.get_shape(), (std::vector<int> { 2, 3, 8, 8 }));
    EXPECT_EQ(batch.targets.get_shape(), (std::vector<int> { 2, 2 }));
    EXPECT_EQ(batch.images.get_device(), Device::GPU);
    const std::vector<float> targets = batch.targets.to_host();
    EXPECT_FLOAT_EQ(targets[0] + targets[1], 1.0F);
    EXPECT_FLOAT_EQ(targets[2] + targets[3], 1.0F);
}

TEST_F(ClassificationLoaderTest, CoversEverySampleAcrossPrefetchedBatchesWithoutOverlap)
{
    // Given: A deterministic (unshuffled) loader with batch size 2
    ClassificationLoader loader((root_ / "train").string(), "train", 2, 8, false);

    // When: All batches are drained
    int images_seen = 0;
    int batches = 0;
    while (loader.has_next())
    {
        Batch batch = loader.get_batch();
        synchronize_device();
        images_seen += batch.images.get_shape()[0];
        ++batches;
    }

    // Then: Three batches cover five images exactly once (2 + 2 + 1)
    EXPECT_EQ(batches, 3);
    EXPECT_EQ(images_seen, 5);
    EXPECT_FALSE(loader.has_next());
    EXPECT_THROW(static_cast<void>(loader.get_batch()), std::runtime_error);
}

TEST_F(ClassificationLoaderTest, ResetRestartsEpochAndAllowsReuse)
{
    // Given: A loader that has already been exhausted
    ClassificationLoader loader((root_ / "train").string(), "train", 4, 8, false);
    while (loader.has_next())
    {
        static_cast<void>(loader.get_batch());
        synchronize_device();
    }

    // When: reset() is called
    loader.reset();

    // Then: The epoch can be iterated again
    EXPECT_TRUE(loader.has_next());
    Batch batch = loader.get_batch();
    synchronize_device();
    EXPECT_EQ(batch.images.get_shape()[0], 4);
}

TEST_F(ClassificationLoaderTest, LockedClassNamesKeepOneHotWidthWhenAFolderIsMissing)
{
    // Given: Train has cat+dog, but the test split is missing the cat folder
    write_solid_png(root_ / "test" / "dog" / "t.png", 0, 0, 255);
    ClassificationLoader train_loader((root_ / "train").string(), "train", 2, 8, false);

    // When: Test reuses the train class vocabulary
    ClassificationLoader test_loader(
        (root_ / "test").string(), "test", 1, 8, false, train_loader.class_names());
    Batch batch = test_loader.get_batch();
    synchronize_device();

    // Then: Targets stay 2-way one-hot even though only one test folder exists
    EXPECT_EQ(test_loader.num_classes(), 2);
    EXPECT_EQ(test_loader.size(), 1U);
    EXPECT_EQ(batch.targets.get_shape(), (std::vector<int> { 1, 2 }));
    const std::vector<float> targets = batch.targets.to_host();
    EXPECT_FLOAT_EQ(targets[0], 0.0F);
    EXPECT_FLOAT_EQ(targets[1], 1.0F);
}

TEST_F(ClassificationLoaderTest, RejectsInvalidConstructorArguments)
{
    // Given: A valid dataset root
    const std::string root = (root_ / "train").string();

    // When / Then: Non-positive batch or image size throw before I/O
    EXPECT_THROW(ClassificationLoader(root, "train", 0, 8, false), std::runtime_error);
    EXPECT_THROW(ClassificationLoader(root, "train", 2, 0, false), std::runtime_error);
    EXPECT_THROW(ClassificationLoader((root_ / "does-not-exist").string(), "train", 2, 8, false), std::runtime_error);
}

TEST(ParallelForCpuTest, ExecutesEveryIndexExactlyOnce)
{
    // Given: A shared histogram protected by a mutex
    constexpr int kCount = 64;
    std::vector<int> hits(static_cast<std::size_t>(kCount), 0);
    std::mutex mutex;

    // When: parallel_for visits every index
    dl::parallel_for(kCount,
        [&](int index)
        {
            std::lock_guard<std::mutex> lock(mutex);
            hits[static_cast<std::size_t>(index)] += 1;
        });

    // Then: Each index ran once and worker count is bounded
    EXPECT_EQ(dl::parallel_worker_count(1), 1);
    EXPECT_LE(dl::parallel_worker_count(128), 16);
    for (int hit : hits)
    {
        EXPECT_EQ(hit, 1);
    }
}

TEST(ParallelForCpuTest, PropagatesWorkerExceptions)
{
    // Given: A functor that throws on one index
    // When / Then: parallel_for rethrows after joining workers
    EXPECT_THROW(dl::parallel_for(8,
                      [](int index)
                      {
                          if (index == 3)
                          {
                              throw std::runtime_error("worker boom");
                          }
                      }),
        std::runtime_error);
}
