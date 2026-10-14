#include "test_helpers.hpp"

#include "DeepLearnLib/core/Network.hpp"
#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/layers/Flatten.hpp"
#include "DeepLearnLib/layers/FullyConnected.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

class NetworkTest : public GpuTest
{
};

TEST_F(NetworkTest, NullLayerThrows)
{
    // Given: A layer list containing a null pointer
    std::vector<std::shared_ptr<Layer>> layers { nullptr };

    // When: The network is constructed
    // Then: Construction throws
    EXPECT_THROW(Network(layers, 0.001F), std::runtime_error);
}

TEST_F(NetworkTest, ForwardThroughFlattenAndDense)
{
    // Given: Flatten followed by a 4->3 dense layer with known weights
    auto flatten = std::make_shared<Flatten>();
    auto dense = std::make_shared<FullyConnected>(4, 3, 0.0F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 4, 3 },
        { 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F });
    set_named_parameter(params, "bias", { 1, 3 }, { 0.0F, 0.0F, 0.0F });
    dense->set_parameters(params);
    Network network({ flatten, dense }, 0.01F);
    Tensor input = Tensor::from_host({ 1, 1, 2, 2 }, { 1.0F, 2.0F, 3.0F, 4.0F }, Device::GPU);

    // When: The network forward pass is computed
    Tensor output = network.forward(input);
    synchronize_device();

    // Then: The output is the first three flattened features
    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 3 }));
    expect_near_vector(output.to_host(), { 1.0F, 2.0F, 3.0F });
}

TEST_F(NetworkTest, SaveLoadRoundTripRestoresWeights)
{
    // Given: A dense network with known weights
    auto dense = std::make_shared<FullyConnected>(2, 2, 0.0F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 2, 2 }, { 0.1F, 0.2F, 0.3F, 0.4F });
    set_named_parameter(params, "bias", { 1, 2 }, { 0.5F, 0.6F });
    dense->set_parameters(params);
    Network network({ dense }, 0.01F);
    const std::string path = "dllib_test_network_weights.bin";

    // When: Weights are saved and loaded into a fresh network
    network.save(path);
    auto loaded_dense = std::make_shared<FullyConnected>(2, 2, 0.0F);
    Network loaded({ loaded_dense }, 0.01F);
    loaded.load(path);
    std::remove(path.c_str());

    // Then: The restored parameters match
    auto restored = loaded_dense->get_parameters();
    expect_near_vector(restored.at("weights").to_host(), { 0.1F, 0.2F, 0.3F, 0.4F });
    expect_near_vector(restored.at("bias").to_host(), { 0.5F, 0.6F });
}

TEST_F(NetworkTest, LoadMissingFileThrows)
{
    // Given: A network and a path that does not exist
    auto dense = std::make_shared<FullyConnected>(2, 2);
    Network network({ dense }, 0.01F);

    // When: load is called
    // Then: Opening the file throws
    EXPECT_THROW(network.load("this_model_file_does_not_exist.pt"), std::runtime_error);
}

TEST_F(NetworkTest, LoadLayerCountMismatchThrows)
{
    // Given: A one-layer checkpoint loaded into a two-layer network
    auto dense = std::make_shared<FullyConnected>(2, 2);
    Network one_layer({ dense }, 0.01F);
    const std::string path = "dllib_test_network_mismatch.bin";
    one_layer.save(path);
    auto flatten = std::make_shared<Flatten>();
    auto dense2 = std::make_shared<FullyConnected>(2, 2);
    Network two_layers({ flatten, dense2 }, 0.01F);

    // When: The checkpoint is loaded
    // Then: The layer-count check throws
    EXPECT_THROW(two_layers.load(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST_F(NetworkTest, AssignsLearningRateToLayers)
{
    // Given: A dense layer with a default learning rate
    auto dense = std::make_shared<FullyConnected>(2, 2);
    const float learning_rate = 0.05F;

    // When: The network is constructed
    Network network({ dense }, learning_rate);

    // Then: The layer learning rate is overwritten
    EXPECT_FLOAT_EQ(dense->learning_rate, learning_rate);
}
