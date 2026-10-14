#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "models/FusedCbrCNN.hpp"

#include <vector>

using namespace dl;
using namespace dllib_test;

class FusedCbrCnnTest : public GpuTest
{
};

TEST_F(FusedCbrCnnTest, LogitsMatchClassCount)
{
    constexpr int batch = 2;
    constexpr int num_classes = 10;
    constexpr int image_size = 32;
    FusedCbrCNN model(num_classes, image_size, 3);
    for (auto& layer : model.get_all_layers())
    {
        layer->to(Device::GPU);
        layer->eval();
    }
    Tensor input = Tensor::from_host({ batch, 3, image_size, image_size },
        std::vector<float>(static_cast<std::size_t>(batch * 3 * image_size * image_size), 0.25F), Device::GPU);

    Tensor logits = model.forward_logits(input);
    synchronize_device();

    EXPECT_EQ(model.get_all_layers().size(), 6U);
    EXPECT_EQ(model.num_classes(), num_classes);
    EXPECT_EQ(model.image_size(), image_size);
    EXPECT_EQ(model.in_channels(), 3);
    EXPECT_EQ(logits.get_shape(), (std::vector<int> { batch, num_classes }));
    expect_all_finite(logits.to_host());
}

TEST(FusedCbrCnnShape, RejectsNonPositiveSizes)
{
    EXPECT_THROW(FusedCbrCNN(0, 32, 3), std::runtime_error);
    EXPECT_THROW(FusedCbrCNN(10, 0, 3), std::runtime_error);
    EXPECT_THROW(FusedCbrCNN(10, 32, 0), std::runtime_error);
}
