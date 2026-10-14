#include "test_helpers.hpp"

#include "DeepLearnLib/loaders/Ecg200Loader.hpp"
#include "DeepLearnLib/layers/Gru.hpp"
#include "DeepLearnLib/core/Tensor.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

namespace
{

auto write_text(const std::filesystem::path& path, const std::string& text) -> void
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    ASSERT_TRUE(static_cast<bool>(stream));
    stream << text;
}

} // namespace

TEST(Ecg200Loader, ReadsTextAndTsLabels)
{
    const auto root = std::filesystem::temp_directory_path() / "dllib_ecg200_rows";
    write_text(root / "ECG200_TRAIN.txt", "-1.0000000e+00 0.25 0.50\n1.0000000e+00 -0.25 0.75\n");
    write_text(root / "ECG200_TEST.ts", "@problemName ECG200\n@classLabel true -1 1\n@data\n0.1,0.2,0.3:-1\n0.4,0.5,0.6:1\n");

    const Ecg200Data train = load_ecg200_file(find_ecg200_split(root, "TRAIN"));
    const Ecg200Data test = load_ecg200_file(find_ecg200_split(root, "TEST"));

    EXPECT_EQ(train.size(), 2);
    EXPECT_EQ(train.length, 2);
    EXPECT_EQ(train.labels, (std::vector<int> { 0, 1 }));
    EXPECT_EQ(train.class_names, (std::vector<std::string> { "-1", "1" }));
    EXPECT_NEAR(train.values[0], 0.25F, 1.0e-6F);
    EXPECT_EQ(test.length, 3);
    EXPECT_EQ(test.labels, (std::vector<int> { 0, 1 }));
    EXPECT_NEAR(test.values[0], 0.1F, 1.0e-6F);
    std::filesystem::remove_all(root);
}

TEST(Ecg200Loader, ReadsArchiveWhenPresent)
{
    const std::vector<std::filesystem::path> roots {
        std::filesystem::current_path() / "data" / "ecg200",
        std::filesystem::current_path() / ".." / "data" / "ecg200",
        std::filesystem::current_path() / ".." / ".." / "data" / "ecg200",
    };
    std::filesystem::path root;
    for (const auto& candidate : roots)
    {
        if (std::filesystem::exists(candidate / "ECG200_TRAIN.txt"))
        {
            root = candidate;
            break;
        }
    }
    if (root.empty())
    {
        GTEST_SKIP() << "ECG200 archive is not in data/ecg200";
    }

    const Ecg200Data train = load_ecg200_file(find_ecg200_split(root, "TRAIN"));
    const Ecg200Data test = load_ecg200_file(find_ecg200_split(root, "TEST"));
    EXPECT_EQ(train.size(), 100);
    EXPECT_EQ(test.size(), 100);
    EXPECT_EQ(train.length, 96);
    EXPECT_EQ(test.length, 96);
    EXPECT_EQ(train.class_names, test.class_names);
}

TEST(GruShape, RejectsEmptySizes)
{
    EXPECT_THROW(Gru(0, 4), std::runtime_error);
    EXPECT_THROW(Gru(4, 0), std::runtime_error);
}

class GruGpuTest : public GpuTest
{
};

TEST_F(GruGpuTest, FinalStateShapeAndWeightGradient)
{
    // Given: A tiny cuDNN GRU and a short batch
    Gru gru(1, 4);
    gru.weight_decay = 0.0F;
    gru.learning_rate = 1.0F;
    gru.use_adam = false;
    const std::vector<float> host_input { 0.2F, -0.4F, 0.6F, 0.1F, -0.2F, 0.3F };
    Tensor input = Tensor::from_host({ 2, 3, 1 }, host_input, Device::GPU);
    const auto before = gru.get_parameters().at("weight_space").to_host();

    // When: The loss is the sum of the final hidden state and one SGD step is taken
    Tensor hidden = gru.forward(input);
    synchronize_device();
    ASSERT_EQ(hidden.get_shape(), (std::vector<int> { 2, 4 }));
    expect_all_finite(hidden.to_host());
    std::vector<float> ones(hidden.get_size(), 1.0F);
    Tensor grad_hidden = Tensor::from_host({ 2, 4 }, ones, Device::GPU);
    Tensor grad_input = gru.backward(grad_hidden);
    synchronize_device();
    EXPECT_EQ(grad_input.get_shape(), (std::vector<int> { 2, 3, 1 }));
    gru.step();
    synchronize_device();
    const auto after = gru.get_parameters().at("weight_space").to_host();

    // Then: At least one cuDNN weight moved, and that partial matches a centered difference
    ASSERT_EQ(before.size(), after.size());
    int pivot = 0;
    float pivot_grad = 0.0F;
    for (std::size_t index = 0; index < before.size(); ++index)
    {
        const float grad = before[index] - after[index];
        if (std::fabs(grad) > std::fabs(pivot_grad))
        {
            pivot_grad = grad;
            pivot = static_cast<int>(index);
        }
    }
    ASSERT_GT(std::fabs(pivot_grad), 1.0e-6F);

    auto loss_of = [&](const std::vector<float>& weights) {
        std::map<std::string, Tensor> params;
        set_named_parameter(params, "weight_space", { static_cast<int>(weights.size()) }, weights);
        gru.set_parameters(params);
        const std::vector<float> values = gru.forward(input).to_host();
        float sum = 0.0F;
        for (float value : values)
        {
            sum += value;
        }
        return sum;
    };
    const float epsilon = 1.0e-3F;
    std::vector<float> plus = before;
    std::vector<float> minus = before;
    plus[static_cast<std::size_t>(pivot)] += epsilon;
    minus[static_cast<std::size_t>(pivot)] -= epsilon;
    const float numeric = (loss_of(plus) - loss_of(minus)) / (2.0F * epsilon);
    const float tolerance = std::max(1.0e-2F, 0.08F * std::fabs(pivot_grad));
    EXPECT_NEAR(numeric, pivot_grad, tolerance);
}
