#include "common/classification_eval.hpp"
#include "common/experiment_config.hpp"
#include "common/forward_plan.hpp"
#include "common/row_batch.hpp"
#include "common/run_metrics.hpp"
#include "common/split_metrics.hpp"
#include "common/tabular_common.hpp"
#include "common/train_loop.hpp"
#include "models/cnn_shapes.hpp"

#include "DeepLearnLib/utils/ExperimentSeed.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <system_error>
#include <vector>

namespace
{

auto vram_for_epoch_test() -> std::size_t
{
    return 64;
}

auto metrics_temp_dir() -> std::filesystem::path
{
    return std::filesystem::temp_directory_path()
        / ("deeplearn_metrics_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

} // namespace

TEST(SplitMetrics, WeightsLossAndAccuracyBySampleCount)
{
    SplitMetrics stats;
    stats.observe(1.0F, 1, 1);
    stats.observe(3.0F, 3, 3);
    stats.observe(9.0F, 0, 0);

    EXPECT_EQ(stats.seen, 4);
    EXPECT_EQ(stats.correct, 4);
    EXPECT_EQ(stats.batches, 2);
    EXPECT_NEAR(stats.loss(), 2.5F, 1.0e-6F);
    EXPECT_NEAR(stats.accuracy(), 1.0F, 1.0e-6F);
}

TEST(SplitMetrics, EmptySplitIsZero)
{
    const SplitMetrics stats;
    EXPECT_FLOAT_EQ(stats.loss(), 0.0F);
    EXPECT_FLOAT_EQ(stats.accuracy(), 0.0F);
}

TEST(TrainTestSplit, SameSeedIsSamePartition)
{
    const IndexSplit first = split_train_test(10, 0.2F, 42U);
    const IndexSplit second = split_train_test(10, 0.2F, 42U);
    EXPECT_EQ(first.train, second.train);
    EXPECT_EQ(first.test, second.test);
    EXPECT_EQ(first.train.size(), 8U);
    EXPECT_EQ(first.test.size(), 2U);

    std::vector<int> seen(10, 0);
    for (const int index : first.train)
    {
        ++seen[static_cast<std::size_t>(index)];
    }
    for (const int index : first.test)
    {
        ++seen[static_cast<std::size_t>(index)];
    }
    EXPECT_EQ(seen, std::vector<int>(10, 1));
}

TEST(TrainTestSplit, RejectsASingleRow)
{
    EXPECT_THROW(split_train_test(1, 0.2F, 1U), std::runtime_error);
    EXPECT_THROW(split_train_test(8, 0.0F, 1U), std::runtime_error);
}

TEST(HostRows, ShapeTailMustMatchWidth)
{
    const std::vector<float> values { 1.0F, 2.0F, 3.0F, 4.0F };
    const std::vector<float> labels { 0.0F, 1.0F };
    HostRows rows;
    rows.values = &values;
    rows.labels = &labels;
    rows.rows = 2;
    rows.width = 2;
    rows.tail = { 2, 2 };
    EXPECT_THROW(require_host_rows(rows), std::runtime_error);
    rows.tail = { 2 };
    EXPECT_NO_THROW(require_host_rows(rows));
    const std::vector<int> shape = batch_shape(2, rows.tail);
    EXPECT_EQ(shape, (std::vector<int> { 2, 2 }));
    EXPECT_EQ(batch_shape_i64(2, rows.tail), (std::vector<int64_t> { 2, 2 }));
}

TEST(MetricsCsv, KeepsSubSecondTime)
{
    const auto directory = metrics_temp_dir();
    std::error_code cleanup_error;
    std::filesystem::remove_all(directory, cleanup_error);
    auto csv = open_metrics_csv(directory, "metrics.csv", kMetricsCsvHeader);
    write_metrics_row(csv, 3, 0.01F, 0.5F, 0.25F, 0.8F, 0.7F, 0.0015, 128);
    write_forward_row(csv, "simplecnn_cifar", 4, 2, 0.5, 64);
    csv.close();

    std::ifstream stream(directory / "metrics.csv");
    std::string header;
    std::string row;
    std::string forward;
    std::getline(stream, header);
    std::getline(stream, row);
    std::getline(stream, forward);
    EXPECT_EQ(header, std::string(kMetricsCsvHeader));
    EXPECT_NE(header.find("TestLoss"), std::string::npos);
    EXPECT_NE(header.find("TestAcc"), std::string::npos);
    EXPECT_NE(header.find("Time(s)"), std::string::npos);
    EXPECT_NE(row.find("0.001500"), std::string::npos);
    EXPECT_EQ(row.find(";2;"), std::string::npos);
    EXPECT_EQ(forward, "simplecnn_cifar;4;2;0.500000;16.000000;64");
    stream.close();
    std::filesystem::remove_all(directory, cleanup_error);
}

TEST(SupervisedEpochs, WritesOneWeightedRowPerEpoch)
{
    const auto directory = metrics_temp_dir();
    std::error_code cleanup_error;
    std::filesystem::remove_all(directory, cleanup_error);
    auto csv = open_metrics_csv(directory, "metrics.csv", kMetricsCsvHeader);
    std::string order;
    run_supervised_epochs(
        2, csv,
        [&](int epoch)
        {
            order += "p" + std::to_string(epoch);
            return epoch == 1 ? 0.1F : 0.01F;
        },
        [&](int epoch)
        {
            order += "t" + std::to_string(epoch);
            SplitMetrics stats;
            stats.observe(epoch == 1 ? 1.0F : 2.0F, 1, 2);
            return stats;
        },
        [&](int epoch)
        {
            order += "e" + std::to_string(epoch);
            SplitMetrics stats;
            stats.observe(4.0F, 1, 4);
            return stats;
        },
        &vram_for_epoch_test);
    csv.close();

    std::ifstream stream(directory / "metrics.csv");
    std::string header;
    std::string first;
    std::string second;
    std::getline(stream, header);
    std::getline(stream, first);
    std::getline(stream, second);
    EXPECT_EQ(order, "p1t1e1p2t2e2");
    EXPECT_EQ(header, std::string(kMetricsCsvHeader));
    EXPECT_NE(first.find("1;0.100000;1.000000;4.000000;0.500000;0.250000;"), std::string::npos);
    EXPECT_NE(second.find("2;0.010000;2.000000;4.000000;0.500000;0.250000;"), std::string::npos);
    EXPECT_EQ(first.substr(first.rfind(';') + 1), "64");
    const auto column = [](const std::string& row, int index) -> std::string
    {
        std::string cell;
        int seen = 0;
        for (const char ch : row)
        {
            if (ch == ';')
            {
                if (seen == index)
                {
                    return cell;
                }
                cell.clear();
                ++seen;
                continue;
            }
            cell.push_back(ch);
        }
        return seen == index ? cell : std::string {};
    };
    const std::regex microseconds(R"(\d+\.\d{6})");
    EXPECT_TRUE(std::regex_match(column(first, 6), microseconds));
    EXPECT_TRUE(std::regex_match(column(second, 6), microseconds));
    stream.close();
    std::filesystem::remove_all(directory, cleanup_error);
}

TEST(ForwardPlan, BothStacksReadTheSameCases)
{
    const ForwardPlan plan = load_forward_plan();
    ASSERT_EQ(plan.cases.size(), 5U);
    EXPECT_EQ(plan.warmup, 10);
    const std::vector<std::string> names
        = { "simplecnn_cifar", "simplecnn_mnist", "tabular_mlp", "gru_classifier", "fused_cbr_cnn" };
    const std::vector<std::string> kinds = { "simplecnn", "simplecnn", "tabular", "gru", "fused" };
    for (std::size_t index = 0; index < plan.cases.size(); ++index)
    {
        EXPECT_EQ(plan.cases[index].name, names[index]);
        EXPECT_EQ(plan.cases[index].kind, kinds[index]);
        EXPECT_GE(plan.cases[index].batch, 1);
        EXPECT_GE(plan.cases[index].repeats, 1);
    }
}

TEST(ExperimentSeed, ShuffleRepeatsForTheSameSeed)
{
    auto draw = []()
    {
        std::vector<int> values(10);
        std::iota(values.begin(), values.end(), 0);
        std::mt19937 rng(dl::kExperimentSeed);
        dl::portable_shuffle(values, rng);
        return values;
    };
    const std::vector<int> first = draw();
    EXPECT_EQ(first, draw());
    EXPECT_NE(first, (std::vector<int> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));
    EXPECT_FLOAT_EQ(dl::paired_uniform(0, -0.5F, 0.5F, dl::kConvWeightSeed),
        dl::paired_uniform(0, -0.5F, 0.5F, dl::kConvWeightSeed));
}

TEST(LearningRateSchedule, DropsOnTheConfiguredEpochs)
{
    nlohmann::json config;
    config["epochs"] = 10;
    config["learning_rate"] = 0.1F;
    EXPECT_FLOAT_EQ(scheduled_learning_rate(config, 7), 0.1F);
    EXPECT_NEAR(scheduled_learning_rate(config, 8), 0.01F, 1.0e-6F);
    EXPECT_NEAR(scheduled_learning_rate(config, 10), 0.001F, 1.0e-7F);

    config["lr_schedule"] = nlohmann::json::array({ { { "until_epoch", 2 }, { "learning_rate", 0.2 } },
        { { "until_epoch", 4 }, { "learning_rate", 0.02 } } });
    EXPECT_FLOAT_EQ(scheduled_learning_rate(config, 2), 0.2F);
    EXPECT_FLOAT_EQ(scheduled_learning_rate(config, 3), 0.02F);
}

TEST(CnnShapes, TwoPoolsDivideTheSideByFour)
{
    EXPECT_EQ(pooled_cnn_features(32), 32 * 8 * 8);
    EXPECT_EQ(pooled_cnn_features(28), 32 * 7 * 7);
    EXPECT_EQ(format_decimal(1.25, 6), "1.250000");
}

TEST(LabelEncoding, OneHotAndArgmaxAgree)
{
    const std::vector<float> encoded = one_hot_labels({ 0.0F, 2.0F, 1.0F }, 3);
    EXPECT_EQ(encoded, (std::vector<float> { 1, 0, 0, 0, 0, 1, 0, 1, 0 }));
    EXPECT_EQ(classification_argmax_row(encoded, 1, 3), 2);
    EXPECT_EQ(count_correct_labels(encoded, { 0.0F, 2.0F, 0.0F }, 3), 2);
}
