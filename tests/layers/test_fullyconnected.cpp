#include "test_helpers.hpp"

#include "DeepLearnLib/core/Tensor.hpp"
#include "DeepLearnLib/layers/FullyConnected.hpp"

#include <map>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

class FullyConnectedTest : public GpuTest
{
};

TEST_F(FullyConnectedTest, ForwardMatchesMatmulPlusBias)
{

    FullyConnected dense(2, 3, 0.0F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 2, 3 }, { 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F });
    set_named_parameter(params, "bias", { 1, 3 }, { 0.5F, -1.0F, 2.0F });
    dense.set_parameters(params);
    Tensor input = Tensor::from_host({ 1, 2 }, { 1.0F, 2.0F }, Device::GPU);

    Tensor output = dense.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int> { 1, 3 }));
    expect_near_vector(output.to_host(), { 1.5F, 1.0F, 5.0F });
}

TEST_F(FullyConnectedTest, BackwardAndStepUpdateWeights)
{

    FullyConnected dense(2, 2, 0.0F);
    dense.learning_rate = 1.0F;
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 2, 2 }, { 0.0F, 0.0F, 0.0F, 0.0F });
    set_named_parameter(params, "bias", { 1, 2 }, { 0.0F, 0.0F });
    dense.set_parameters(params);
    Tensor input = Tensor::from_host({ 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F }, Device::GPU);

    Tensor output = dense.forward(input);
    synchronize_device();

    expect_near_vector(output.to_host(), { 0.0F, 0.0F, 0.0F, 0.0F });

    Tensor grad_output = Tensor::from_host({ 2, 2 }, { 1.0F, 0.0F, 0.0F, 1.0F }, Device::GPU);
    Tensor grad_input = dense.backward(grad_output);
    synchronize_device();

    EXPECT_EQ(grad_input.get_shape(), (std::vector<int> { 2, 2 }));
    expect_near_vector(grad_input.to_host(), { 0.0F, 0.0F, 0.0F, 0.0F });

    dense.step();
    auto updated = dense.get_parameters();
    synchronize_device();

    expect_near_vector(updated.at("weights").to_host(), { -1.0F, 0.0F, 0.0F, -1.0F }, kLooseEpsilon);
}

TEST_F(FullyConnectedTest, InvalidSizesThrow)
{

    EXPECT_THROW(FullyConnected(0, 4), std::runtime_error);
    EXPECT_THROW(FullyConnected(4, 0), std::runtime_error);
    EXPECT_THROW(FullyConnected(-1, 2), std::runtime_error);
}

TEST_F(FullyConnectedTest, ForwardRejectsWrongRankDeviceAndFeatures)
{

    FullyConnected dense(3, 2);
    Tensor rank1 = Tensor::from_host({ 3 }, { 1.0F, 2.0F, 3.0F }, Device::GPU);
    Tensor wrong_features = Tensor::from_host({ 1, 2 }, { 1.0F, 2.0F }, Device::GPU);
    Tensor cpu = Tensor::from_host({ 1, 3 }, { 1.0F, 2.0F, 3.0F }, Device::CPU);

    EXPECT_THROW(dense.forward(rank1), std::runtime_error);
    EXPECT_THROW(dense.forward(wrong_features), std::runtime_error);
    EXPECT_THROW(dense.forward(cpu), std::runtime_error);
}

TEST_F(FullyConnectedTest, BackwardWithoutForwardThrows)
{

    FullyConnected dense(2, 2);
    Tensor grad = Tensor::from_host({ 1, 2 }, { 1.0F, 1.0F }, Device::GPU);

    EXPECT_THROW(dense.backward(grad), std::runtime_error);
}

TEST_F(FullyConnectedTest, SecondBackwardWithoutForwardThrows)
{

    FullyConnected dense(2, 2);
    Tensor input = Tensor::from_host({ 1, 2 }, { 1.0F, 2.0F }, Device::GPU);
    (void)dense.forward(input);
    Tensor grad = Tensor::from_host({ 1, 2 }, { 1.0F, 1.0F }, Device::GPU);
    (void)dense.backward(grad);

    EXPECT_THROW(dense.backward(grad), std::runtime_error);
}

TEST_F(FullyConnectedTest, BackwardRejectsBatchMismatch)
{

    FullyConnected dense(2, 2);
    Tensor input = Tensor::from_host({ 1, 2 }, { 1.0F, 2.0F }, Device::GPU);
    (void)dense.forward(input);
    Tensor grad = Tensor::from_host({ 2, 2 }, std::vector<float>(4, 1.0F), Device::GPU);

    EXPECT_THROW(dense.backward(grad), std::runtime_error);
}

TEST_F(FullyConnectedTest, ZeroWeightsReturnBias)
{

    FullyConnected dense(2, 2, 0.0F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 2, 2 }, { 0.0F, 0.0F, 0.0F, 0.0F });
    set_named_parameter(params, "bias", { 1, 2 }, { 0.5F, -1.5F });
    dense.set_parameters(params);
    Tensor input = Tensor::from_host({ 3, 2 }, std::vector<float>(6, 9.0F), Device::GPU);

    Tensor output = dense.forward(input);
    synchronize_device();

    EXPECT_EQ(output.get_shape(), (std::vector<int> { 3, 2 }));
    expect_near_vector(output.to_host(), { 0.5F, -1.5F, 0.5F, -1.5F, 0.5F, -1.5F });
}

TEST_F(FullyConnectedTest, ParameterRoundTripAndGpuTo)
{

    FullyConnected dense(2, 1);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { 2, 1 }, { 0.25F, -0.5F });
    set_named_parameter(params, "bias", { 1, 1 }, { 0.1F });
    dense.set_parameters(params);

    auto restored = dense.get_parameters();
    dense.to(Device::GPU);

    expect_near_vector(restored.at("weights").to_host(), { 0.25F, -0.5F });
    expect_near_vector(restored.at("bias").to_host(), { 0.1F });
    EXPECT_THROW(dense.to(Device::CPU), std::runtime_error);
}

TEST_F(FullyConnectedTest, DefaultInitializationIsFinite)
{

    FullyConnected dense(8, 4);

    auto params = dense.get_parameters();

    EXPECT_EQ(params.at("weights").get_shape(), (std::vector<int> { 8, 4 }));
    EXPECT_EQ(params.at("bias").get_shape(), (std::vector<int> { 1, 4 }));
    expect_all_finite(params.at("weights").to_host());
    expect_all_finite(params.at("bias").to_host());
}
