#include "test_helpers.hpp"

#include "SimpleCNN.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <vector>

using namespace dl;
using namespace dllib_test;

class SimpleCnnTest : public GpuTest
{
};

TEST_F(SimpleCnnTest, ForwardLogitsMatchClassCount)
{
    // Given: A 10-class SimpleCNN and a CIFAR-sized NCHW batch
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

    // When: Logits and softmax probabilities are computed
    Tensor logits = model.forward_logits(input);
    Tensor probabilities = model.forward(input);
    synchronize_device();

    // Then: Both outputs are [N, C] GPU tensors
    EXPECT_EQ(logits.get_shape(), (std::vector<int> { batch, num_classes }));
    EXPECT_EQ(probabilities.get_shape(), (std::vector<int> { batch, num_classes }));
    EXPECT_EQ(logits.get_device(), Device::GPU);
    expect_all_finite(logits.to_host());
    expect_all_finite(probabilities.to_host());
}

TEST_F(SimpleCnnTest, TrainableStackHasExpectedDepth)
{
    // Given: A 4-class SimpleCNN
    SimpleCNN model(4, 32);

    // When: Trainable layers are listed
    auto layers = model.get_all_layers();

    // Then: Conv-ReLU-Pool is repeated twice, then Flatten and FullyConnected
    EXPECT_EQ(layers.size(), 8U);
    EXPECT_EQ(model.num_classes(), 4);
    EXPECT_EQ(model.image_size(), 32);
    for (const auto& layer : layers)
    {
        ASSERT_NE(layer, nullptr);
    }
}
