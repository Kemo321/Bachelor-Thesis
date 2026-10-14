#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "models/TabularMLP.hpp"

#include <vector>

using namespace dl;
using namespace dllib_test;

class TabularMlpTest : public GpuTest
{
};

TEST_F(TabularMlpTest, LogitsMatchClassCount)
{
    constexpr int batch = 4;
    constexpr int features = 4;
    constexpr int hidden = 16;
    constexpr int num_classes = 3;
    TabularMLP model(features, hidden, num_classes);
    for (auto& layer : model.get_all_layers())
    {
        layer->to(Device::GPU);
        layer->eval();
    }
    Tensor input = Tensor::from_host({ batch, features },
        std::vector<float>(static_cast<std::size_t>(batch * features), 0.25F), Device::GPU);

    Tensor logits = model.forward_logits(input);
    synchronize_device();

    EXPECT_EQ(model.get_all_layers().size(), 3U);
    EXPECT_EQ(model.in_features(), features);
    EXPECT_EQ(model.hidden_size(), hidden);
    EXPECT_EQ(model.num_classes(), num_classes);
    EXPECT_EQ(logits.get_shape(), (std::vector<int> { batch, num_classes }));
    expect_all_finite(logits.to_host());
}

TEST(TabularMlp, RejectsNonPositiveSizes)
{
    EXPECT_THROW(TabularMLP(0, 16, 3), std::runtime_error);
    EXPECT_THROW(TabularMLP(4, 0, 3), std::runtime_error);
    EXPECT_THROW(TabularMLP(4, 16, 0), std::runtime_error);
}
