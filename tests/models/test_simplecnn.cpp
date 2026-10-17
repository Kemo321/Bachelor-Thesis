#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "models/SimpleCNN.hpp"

#include <vector>

using namespace dl;
using namespace dllib_test;

class SimpleCnnTest : public GpuTest
{
};

TEST_F(SimpleCnnTest, ForwardLogitsMatchClassCount)
{

    constexpr int batch = 2;
    constexpr int num_classes = 10;
    constexpr int image_size = 32;
    SimpleCNN model(num_classes, image_size);
    for (auto& layer : model.get_all_layers())
    {
        layer->to(Device::GPU);
        layer->eval();
    }
    Tensor input = Tensor::from_host({ batch, 3, image_size, image_size },
        std::vector<float>(static_cast<std::size_t>(batch * 3 * image_size * image_size), 0.25F), Device::GPU);

    Tensor logits = model.forward_logits(input);
    synchronize_device();

    EXPECT_EQ(logits.get_shape(), (std::vector<int> { batch, num_classes }));
    EXPECT_EQ(logits.get_device(), Device::GPU);
    expect_all_finite(logits.to_host());
}

TEST_F(SimpleCnnTest, TrainableStackHasExpectedDepth)
{

    SimpleCNN model(4, 32);

    auto layers = model.get_all_layers();

    EXPECT_EQ(layers.size(), 8U);
    EXPECT_EQ(model.num_classes(), 4);
    EXPECT_EQ(model.image_size(), 32);
    EXPECT_EQ(model.in_channels(), 3);
    for (const auto& layer : layers)
    {
        ASSERT_NE(layer, nullptr);
    }
}

TEST_F(SimpleCnnTest, AcceptsSingleChannelMnistInput)
{
    constexpr int batch = 2;
    constexpr int num_classes = 10;
    constexpr int image_size = 28;
    SimpleCNN model(num_classes, image_size, 1);
    for (auto& layer : model.get_all_layers())
    {
        layer->to(Device::GPU);
        layer->eval();
    }
    Tensor input = Tensor::from_host({ batch, 1, image_size, image_size },
        std::vector<float>(static_cast<std::size_t>(batch * 1 * image_size * image_size), 0.5F), Device::GPU);
    Tensor logits = model.forward_logits(input);
    synchronize_device();
    EXPECT_EQ(model.in_channels(), 1);
    EXPECT_EQ(logits.get_shape(), (std::vector<int> { batch, num_classes }));
    expect_all_finite(logits.to_host());
}
