#include "test_helpers.hpp"

#include "DeepLearnLib/DarknetDetectionLoss.hpp"
#include "DeepLearnLib/DarknetWeights.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/FusedCBR2d.hpp"
#include "DeepLearnLib/LocalLayer.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

namespace
{

auto unique_weights_path() -> std::filesystem::path
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / ("dllib_darknet_" + std::to_string(stamp) + ".weights");
}

auto write_i32(std::ofstream& stream, int value) -> void
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(int));
}

auto write_u64(std::ofstream& stream, std::uint64_t value) -> void
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(std::uint64_t));
}

auto write_f32(std::ofstream& stream, const std::vector<float>& values) -> void
{
    stream.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
}

} // namespace

class DarknetTest : public GpuTest
{
};

TEST_F(DarknetTest, LocalLayerMatchesDenseOneByOne)
{
    LocalLayer layer(1, 1, 1, 1, 0, 2, 2);
    std::map<std::string, Tensor> params;
    params.emplace("weights", Tensor::from_host({ 4, 1, 1, 1, 1 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU));
    params.emplace("bias", Tensor::from_host({ 1, 1, 2, 2 }, { 0.1F, 0.2F, 0.3F, 0.4F }, Device::GPU));
    layer.set_parameters(params);

    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);
    Tensor output = layer.forward(input);
    synchronize_device();
    const std::vector<float> host = output.to_host();
    ASSERT_EQ(host.size(), 4U);
    EXPECT_NEAR(host[0], 1.0F * 1.0F + 0.1F, kEpsilon);
    EXPECT_NEAR(host[1], 2.0F * 2.0F + 0.2F, kEpsilon);
    EXPECT_NEAR(host[2], 3.0F * 3.0F + 0.3F, kEpsilon);
    EXPECT_NEAR(host[3], 4.0F * 4.0F + 0.4F, kEpsilon);
}

TEST_F(DarknetTest, LoadsFusedCbrAndTransposesConnected)
{
    const auto path = unique_weights_path();
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream);
    write_i32(stream, 0);
    write_i32(stream, 2);
    write_i32(stream, 0);
    write_u64(stream, 123);
    write_f32(stream, { 0.1F, 0.2F, 0.3F, 0.4F });
    write_f32(stream, { 1.1F, 1.2F, 1.3F, 1.4F });
    write_f32(stream, { 0.0F, 0.0F, 0.0F, 0.0F });
    write_f32(stream, { 1.0F, 1.0F, 1.0F, 1.0F });
    write_f32(stream, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F });
    write_f32(stream, { 0.5F, 0.6F });
    write_f32(stream, { 10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F });
    stream.close();

    auto conv = std::make_shared<FusedCBR2d>(3, 4, 1, 1, 0);
    auto fc = std::make_shared<FullyConnected>(3, 2);
    const auto report = load_darknet_weights({ conv, fc }, path.string());
    EXPECT_EQ(report.convs_loaded, 1);
    EXPECT_EQ(report.connected_loaded, 1);
    EXPECT_EQ(report.bytes_remaining, 0U);
    EXPECT_EQ(report.seen, 123U);

    auto conv_params = conv->get_parameters();
    const std::vector<float> gamma = conv_params.at("gamma").to_host();
    const std::vector<float> beta = conv_params.at("beta").to_host();
    const std::vector<float> weights = conv_params.at("weights").to_host();
    EXPECT_NEAR(gamma[0], 1.1F, kEpsilon);
    EXPECT_NEAR(beta[1], 0.2F, kEpsilon);
    EXPECT_NEAR(weights[0], 1.0F, kEpsilon);
    EXPECT_NEAR(weights[11], 12.0F, kEpsilon);

    auto fc_params = fc->get_parameters();
    const std::vector<float> fc_w = fc_params.at("weights").to_host();
    ASSERT_EQ(fc_w.size(), 6U);
    EXPECT_NEAR(fc_w[0], 10.0F, kEpsilon);
    EXPECT_NEAR(fc_w[1], 40.0F, kEpsilon);
    EXPECT_NEAR(fc_w[2], 20.0F, kEpsilon);
    EXPECT_NEAR(fc_w[3], 50.0F, kEpsilon);
    EXPECT_NEAR(fc_w[4], 30.0F, kEpsilon);
    EXPECT_NEAR(fc_w[5], 60.0F, kEpsilon);
    std::filesystem::remove(path);
}

TEST_F(DarknetTest, DetectionLossEmptyCellsIsFinite)
{
    DarknetDetectionLoss::Config config;
    config.num_classes = 2;
    const int pred_len = 7 * 7 * (2 + 3 + 12);
    const int truth_len = 7 * 7 * (1 + 4 + 2);
    Tensor pred = Tensor::from_host({ 1, pred_len }, std::vector<float>(static_cast<std::size_t>(pred_len), 0.0F),
        Device::GPU);
    Tensor tgt = Tensor::from_host({ 1, 7, 7, 7 }, std::vector<float>(static_cast<std::size_t>(truth_len), 0.0F),
        Device::GPU);
    Tensor loss = DarknetDetectionLoss::loss(tgt, pred, config);
    Tensor grad = DarknetDetectionLoss::loss_derivative(tgt, pred, config);
    synchronize_device();
    expect_all_finite(loss.to_host());
    expect_all_finite(grad.to_host());
    EXPECT_EQ(static_cast<int>(grad.get_size()), pred_len);
}
