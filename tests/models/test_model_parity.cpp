#include "test_helpers.hpp"

#include "DeepLearnLib/core/Layer.hpp"
#include "DeepLearnLib/core/Tensor.hpp"
#include "models/FusedCbrCNN.hpp"
#include "models/GruClassifier.hpp"
#include "models/SimpleCNN.hpp"
#include "models/TabularMLP.hpp"
#include "models/torch/TorchFusedCbrCNN.hpp"
#include "models/torch/TorchGruClassifier.hpp"
#include "models/torch/TorchSimpleCNN.hpp"
#include "models/torch/TorchTabularMLP.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace dl;
using namespace dllib_test;

namespace
{

auto custom_parameter_count(const std::vector<std::shared_ptr<Layer>>& layers, bool include_running) -> int64_t
{
    int64_t total = 0;
    for (const auto& layer : layers)
    {
        for (const auto& entry : layer->get_parameters())
        {
            if (!include_running && (entry.first == "running_mean" || entry.first == "running_var"))
            {
                continue;
            }
            total += static_cast<int64_t>(entry.second.get_size());
        }
    }
    return total;
}

template <typename Module>
auto torch_parameter_count(const Module& model) -> int64_t
{
    int64_t total = 0;
    for (const auto& parameter : model->parameters())
    {
        total += parameter.numel();
    }
    return total;
}

auto torch_f32(const torch::Tensor& tensor) -> std::vector<float>
{
    const auto cpu = tensor.detach().contiguous().to(torch::kCPU).to(torch::kFloat32);
    std::vector<float> host(static_cast<std::size_t>(cpu.numel()));
    std::memcpy(host.data(), cpu.data_ptr<float>(), host.size() * sizeof(float));
    return host;
}

auto expect_same_values(const std::vector<float>& custom, const std::vector<float>& reference) -> void
{
    ASSERT_EQ(custom.size(), reference.size());
    for (std::size_t index = 0; index < custom.size(); ++index)
    {
        EXPECT_FLOAT_EQ(custom[index], reference[index]) << "index " << index;
    }
}

auto expect_close_logits(const std::vector<float>& custom, const std::vector<float>& reference, float tolerance) -> void
{
    ASSERT_EQ(custom.size(), reference.size());
    for (std::size_t index = 0; index < custom.size(); ++index)
    {
        EXPECT_NEAR(custom[index], reference[index], tolerance) << "logit " << index;
    }
}

auto expect_transposed_linear(const std::vector<float>& custom_in_out, const torch::Tensor& weight) -> void
{
    const int out_features = static_cast<int>(weight.size(0));
    const int in_features = static_cast<int>(weight.size(1));
    const std::vector<float> reference = torch_f32(weight);
    ASSERT_EQ(custom_in_out.size(), reference.size());
    for (int in = 0; in < in_features; ++in)
    {
        for (int out = 0; out < out_features; ++out)
        {
            const std::size_t custom_index
                = (static_cast<std::size_t>(in) * static_cast<std::size_t>(out_features)) + static_cast<std::size_t>(out);
            const std::size_t torch_index
                = (static_cast<std::size_t>(out) * static_cast<std::size_t>(in_features)) + static_cast<std::size_t>(in);
            EXPECT_FLOAT_EQ(custom_in_out[custom_index], reference[torch_index]);
        }
    }
}

} // namespace

class ModelParityTest : public GpuTest
{
};

TEST_F(ModelParityTest, SimpleCnnParameterCountMatchesTorch)
{
    SimpleCNN custom(10, 32, 3);
    TorchSimpleCNN reference(10, 32, 3);
    EXPECT_EQ(custom_parameter_count(custom.get_all_layers(), true), torch_parameter_count(reference));

    SimpleCNN mnist(10, 28, 1);
    TorchSimpleCNN mnist_reference(10, 28, 1);
    EXPECT_EQ(custom_parameter_count(mnist.get_all_layers(), true), torch_parameter_count(mnist_reference));

    auto layers = custom.get_all_layers();
    expect_same_values(layers[0]->get_parameters().at("weights").to_host(), torch_f32(reference->conv1->weight));
    expect_same_values(layers[0]->get_parameters().at("bias").to_host(), torch_f32(reference->conv1->bias));
    expect_same_values(layers[3]->get_parameters().at("weights").to_host(), torch_f32(reference->conv2->weight));
    expect_transposed_linear(layers.back()->get_parameters().at("weights").to_host(), reference->fc->weight);
    expect_same_values(layers.back()->get_parameters().at("bias").to_host(), torch_f32(reference->fc->bias));

    for (auto& layer : layers)
    {
        layer->to(Device::GPU);
        layer->eval();
    }
    Tensor input = Tensor::from_host({ 1, 3, 32, 32 }, std::vector<float>(3U * 32U * 32U, 0.25F), Device::GPU);
    Tensor logits = custom.forward_logits(input);
    synchronize_device();
    reference->eval();
    const torch::Device device(torch::kCUDA);
    reference->to(device);
    const torch::Tensor torch_logits = reference->forward(torch::full(
        { 1, 3, 32, 32 }, 0.25, torch::TensorOptions().dtype(torch::kFloat32).device(device)));
    expect_close_logits(logits.to_host(), torch_f32(torch_logits), 1.0e-4F);
}

TEST_F(ModelParityTest, TabularMlpParameterCountMatchesTorch)
{
    TabularMLP custom(4, 16, 3);
    TorchTabularMLP reference(4, 16, 3);
    EXPECT_EQ(custom_parameter_count(custom.get_all_layers(), true), torch_parameter_count(reference));
    auto layers = custom.get_all_layers();
    expect_transposed_linear(layers[0]->get_parameters().at("weights").to_host(), reference->fc1->weight);
    expect_same_values(layers[0]->get_parameters().at("bias").to_host(), torch_f32(reference->fc1->bias));
    expect_transposed_linear(layers[2]->get_parameters().at("weights").to_host(), reference->fc2->weight);
    expect_same_values(layers[2]->get_parameters().at("bias").to_host(), torch_f32(reference->fc2->bias));

    for (auto& layer : layers)
    {
        layer->to(Device::GPU);
        layer->eval();
    }
    const std::vector<float> rows { 0.1F, -0.2F, 0.3F, 0.4F, 0.5F, 0.0F, -0.1F, 0.2F };
    Tensor input = Tensor::from_host({ 2, 4 }, rows, Device::GPU);
    Tensor logits = custom.forward_logits(input);
    synchronize_device();
    reference->eval();
    const torch::Device device(torch::kCUDA);
    reference->to(device);
    const torch::Tensor torch_input = torch::from_blob(const_cast<float*>(rows.data()), { 2, 4 }, torch::kFloat32)
                                          .clone()
                                          .to(device);
    expect_close_logits(logits.to_host(), torch_f32(reference->forward(torch_input)), 1.0e-5F);
}

TEST_F(ModelParityTest, FusedCbrTrainableCountMatchesTorch)
{
    FusedCbrCNN custom(10, 32, 3);
    TorchFusedCbrCNN reference(10, 32, 3);
    EXPECT_EQ(custom_parameter_count(custom.get_all_layers(), false), torch_parameter_count(reference));
    EXPECT_GT(custom_parameter_count(custom.get_all_layers(), true), torch_parameter_count(reference));

    auto layers = custom.get_all_layers();
    expect_same_values(layers[0]->get_parameters().at("weights").to_host(), torch_f32(reference->conv1->weight));
    expect_same_values(layers[0]->get_parameters().at("bias").to_host(), torch_f32(reference->conv1->bias));
    expect_same_values(layers[0]->get_parameters().at("gamma").to_host(), torch_f32(reference->bn1->weight));
    expect_same_values(layers[0]->get_parameters().at("beta").to_host(), torch_f32(reference->bn1->bias));
    expect_same_values(layers[2]->get_parameters().at("weights").to_host(), torch_f32(reference->conv2->weight));
    expect_same_values(layers[2]->get_parameters().at("bias").to_host(), torch_f32(reference->conv2->bias));
    expect_same_values(layers[2]->get_parameters().at("gamma").to_host(), torch_f32(reference->bn2->weight));
    expect_same_values(layers[2]->get_parameters().at("beta").to_host(), torch_f32(reference->bn2->bias));
    expect_transposed_linear(layers.back()->get_parameters().at("weights").to_host(), reference->fc->weight);
    expect_same_values(layers.back()->get_parameters().at("bias").to_host(), torch_f32(reference->fc->bias));

    for (auto& layer : layers)
    {
        layer->to(Device::GPU);
        layer->eval();
    }
    Tensor input = Tensor::from_host({ 1, 3, 32, 32 }, std::vector<float>(3U * 32U * 32U, 0.25F), Device::GPU);
    Tensor logits = custom.forward_logits(input);
    synchronize_device();
    reference->eval();
    const torch::Device device(torch::kCUDA);
    reference->to(device);
    const torch::Tensor torch_logits = reference->forward(torch::full(
        { 1, 3, 32, 32 }, 0.25, torch::TensorOptions().dtype(torch::kFloat32).device(device)));
    expect_close_logits(logits.to_host(), torch_f32(torch_logits), 1.0e-4F);
}

TEST_F(ModelParityTest, GruClassifierOutputShapeMatchesTorch)
{
    constexpr int batch = 2;
    constexpr int length = 5;
    constexpr int hidden = 4;
    constexpr int num_classes = 2;
    GruClassifier custom(1, hidden, num_classes);
    for (auto& layer : custom.get_all_layers())
    {
        layer->to(Device::GPU);
        layer->eval();
    }
    Tensor input = Tensor::from_host({ batch, length, 1 },
        std::vector<float>(static_cast<std::size_t>(batch * length), 0.2F), Device::GPU);
    Tensor logits = custom.forward_logits(input);
    synchronize_device();
    EXPECT_EQ(logits.get_shape(), (std::vector<int> { batch, num_classes }));

    TorchGruClassifier reference(1, hidden, num_classes);
    reference->eval();
    const torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    reference->to(device);
    const torch::Tensor torch_logits = reference->forward(torch::full({ batch, length, 1 }, 0.2,
        torch::TensorOptions().dtype(torch::kFloat32).device(device)));
    EXPECT_EQ(torch_logits.size(0), batch);
    EXPECT_EQ(torch_logits.size(1), num_classes);

    const std::vector<float> custom_logits = logits.to_host();
    const std::vector<float> reference_logits = torch_f32(torch_logits);
    ASSERT_EQ(custom_logits.size(), reference_logits.size());
    for (std::size_t index = 0; index < custom_logits.size(); ++index)
    {
        EXPECT_NEAR(custom_logits[index], reference_logits[index], 1.0e-4F) << "logit " << index;
    }
}
