#include "test_helpers.hpp"

#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLO.hpp"

#include <memory>
#include <vector>

using namespace dl;
using namespace dllib_test;

class YoloTest : public GpuTest
{
};

TEST_F(YoloTest, ConstructsNonEmptyBackboneAndHead)
{
    // Given: A default 20-class YOLOv1 model
    YOLO model;

    // When: Layers are inspected
    auto all_layers = model.get_all_layers();

    // Then: Backbone and head are populated and concatenated in order
    EXPECT_FALSE(model.backbone_layers.empty());
    EXPECT_FALSE(model.head_layers.empty());
    EXPECT_EQ(all_layers.size(), model.backbone_layers.size() + model.head_layers.size());
    EXPECT_EQ(all_layers.front().get(), model.backbone_layers.front().get());
    EXPECT_EQ(all_layers.back().get(), model.head_layers.back().get());
    for (const auto& layer : all_layers)
    {
        ASSERT_NE(layer, nullptr);
    }
}

TEST_F(YoloTest, CustomClassCountChangesOutputWidth)
{
    // Given: A 2-class YOLOv1 model and a 448x448 image
    constexpr int num_classes = 2;
    YOLO model(num_classes);
    Tensor input = Tensor::from_host({ 1, 3, 448, 448 }, std::vector<float>(1 * 3 * 448 * 448, 0.5F), Device::GPU);

    // When: The forward pass is computed
    Tensor output = model.forward(input);
    synchronize_device();

    // Then: The flattened detection tensor has width 7*7*(10+2)
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 7 * 7 * (10 + num_classes) }));
    EXPECT_EQ(output.get_device(), Device::GPU);
    expect_all_finite(output.to_host());
}

TEST_F(YoloTest, EvalModeForwardIsFinite)
{
    // Given: A 20-class model switched to eval (disables dropout)
    YOLO model(20);
    for (auto& layer : model.get_all_layers())
    {
        layer->eval();
    }
    Tensor input = Tensor::from_host({ 1, 3, 448, 448 }, std::vector<float>(1 * 3 * 448 * 448, 0.0F), Device::GPU);

    // When: The forward pass is computed
    Tensor output = model.forward(input);
    synchronize_device();

    // Then: The VOC-sized detection tensor is finite
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 7 * 7 * 30 }));
    expect_all_finite(output.to_host());
}
