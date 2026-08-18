#include "test_helpers.hpp"

#include "DeepLearnLib/PackedImageLoader.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

namespace
{

auto unique_bin_path() -> std::filesystem::path
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / ("dllib_mnist_" + std::to_string(stamp) + ".bin");
}

auto write_u32_le(std::ofstream& stream, std::uint32_t value) -> void
{
    const std::array<unsigned char, 4> bytes { static_cast<unsigned char>(value & 0xFFU),
        static_cast<unsigned char>((value >> 8) & 0xFFU), static_cast<unsigned char>((value >> 16) & 0xFFU),
        static_cast<unsigned char>((value >> 24) & 0xFFU) };
    stream.write(reinterpret_cast<const char*>(bytes.data()), 4);
}

auto write_tiny_packed(const std::filesystem::path& path) -> void
{
    constexpr int n = 4;
    constexpr int channels = 1;
    constexpr int height = 2;
    constexpr int width = 2;
    constexpr int classes = 2;
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Could not write packed test file");
    }
    stream.write("DLIMG001", 8);
    write_u32_le(stream, static_cast<std::uint32_t>(n));
    write_u32_le(stream, static_cast<std::uint32_t>(channels));
    write_u32_le(stream, static_cast<std::uint32_t>(height));
    write_u32_le(stream, static_cast<std::uint32_t>(width));
    write_u32_le(stream, static_cast<std::uint32_t>(classes));
    const std::array<unsigned char, 16> pixels { 0, 0, 0, 0, 255, 255, 255, 255, 0, 255, 0, 255, 255, 0, 255, 0 };
    const std::array<unsigned char, 4> labels { 0, 1, 0, 1 };
    stream.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    stream.write(reinterpret_cast<const char*>(labels.data()), static_cast<std::streamsize>(labels.size()));
}

} // namespace

class PackedImageLoaderTest : public GpuTest
{
protected:
    std::filesystem::path path_;

    void SetUp() override
    {
        GpuTest::SetUp();
        if (IsSkipped())
        {
            return;
        }
        path_ = unique_bin_path();
        write_tiny_packed(path_);
    }

    void TearDown() override
    {
        if (!path_.empty())
        {
            std::error_code error;
            std::filesystem::remove(path_, error);
        }
    }
};

TEST_F(PackedImageLoaderTest, YieldsNchwFloatAndOneHotLabels)
{
    PackedImageLoader loader(path_.string(), 2, false);
    EXPECT_EQ(loader.size(), 4U);
    EXPECT_EQ(loader.channels(), 1);
    EXPECT_EQ(loader.height(), 2);
    EXPECT_EQ(loader.width(), 2);
    EXPECT_EQ(loader.num_classes(), 2);

    Batch batch = loader.get_batch();
    EXPECT_EQ(batch.images.get_shape(), (std::vector<int> { 2, 1, 2, 2 }));
    EXPECT_EQ(batch.targets.get_shape(), (std::vector<int> { 2, 2 }));
    const std::vector<float> images = batch.images.to_host();
    const std::vector<float> targets = batch.targets.to_host();
    EXPECT_NEAR(images[0], 0.0F, 1.0e-5F);
    EXPECT_NEAR(images[4], 1.0F, 1.0e-5F);
    EXPECT_NEAR(targets[0], 1.0F, 1.0e-5F);
    EXPECT_NEAR(targets[1], 0.0F, 1.0e-5F);
    EXPECT_NEAR(targets[2], 0.0F, 1.0e-5F);
    EXPECT_NEAR(targets[3], 1.0F, 1.0e-5F);
}

TEST_F(PackedImageLoaderTest, RejectsBadMagic)
{
    const auto bad = std::filesystem::temp_directory_path() / "dllib_bad_magic.bin";
    {
        std::ofstream stream(bad, std::ios::binary);
        stream.write("NOPE!!!!", 8);
    }
    EXPECT_THROW(PackedImageLoader(bad.string(), 1, false), std::runtime_error);
    std::error_code error;
    std::filesystem::remove(bad, error);
}
