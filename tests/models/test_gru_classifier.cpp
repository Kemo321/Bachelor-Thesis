#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "models/GruClassifier.hpp"

#include <vector>

using namespace dl;
using namespace dllib_test;

class GruClassifierTest : public GpuTest
{
};

TEST_F(GruClassifierTest, FinalHiddenFeedsTheClassHead)
{
    constexpr int batch = 2;
    constexpr int length = 5;
    constexpr int input_size = 1;
    constexpr int hidden = 4;
    constexpr int num_classes = 2;
    GruClassifier model(input_size, hidden, num_classes);
    for (auto& layer : model.get_all_layers())
    {
        layer->to(Device::GPU);
        layer->eval();
    }
    Tensor input = Tensor::from_host({ batch, length, input_size },
        std::vector<float>(static_cast<std::size_t>(batch * length * input_size), 0.1F), Device::GPU);

    Tensor logits = model.forward_logits(input);
    synchronize_device();

    EXPECT_EQ(model.get_all_layers().size(), 2U);
    EXPECT_EQ(model.input_size(), input_size);
    EXPECT_EQ(model.hidden_size(), hidden);
    EXPECT_EQ(model.num_classes(), num_classes);
    EXPECT_EQ(logits.get_shape(), (std::vector<int> { batch, num_classes }));
    expect_all_finite(logits.to_host());
}

TEST(GruClassifierShape, RejectsNonPositiveSizes)
{
    EXPECT_THROW(GruClassifier(0, 4, 2), std::runtime_error);
    EXPECT_THROW(GruClassifier(1, 0, 2), std::runtime_error);
    EXPECT_THROW(GruClassifier(1, 4, 0), std::runtime_error);
}
