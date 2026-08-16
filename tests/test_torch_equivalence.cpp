#include "test_helpers.hpp"

#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Tensor.hpp"

#include <cmath>
#include <map>
#include <random>
#include <string>
#include <torch/torch.h>
#include <vector>

using namespace dl;
using namespace dllib_test;

namespace
{

constexpr float kTorchTol = 1e-4F;

auto random_host(std::size_t count, unsigned seed, float scale = 0.5F) -> std::vector<float>
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> values(count);
    for (float& value : values)
    {
        value = dist(rng);
    }
    return values;
}

auto unique_host(const std::vector<int>& shape, unsigned seed) -> std::vector<float>
{
    std::size_t count = 1;
    for (int dimension : shape)
    {
        count *= static_cast<std::size_t>(dimension);
    }
    std::vector<float> values = random_host(count, seed, 0.25F);
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        values[index] += static_cast<float>(index) * 1.0e-3F;
    }
    return values;
}

auto torch_from_host(const std::vector<int64_t>& sizes, const std::vector<float>& host, bool requires_grad)
    -> torch::Tensor
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto cpu = torch::from_blob(const_cast<float*>(host.data()), sizes, options).clone();
    auto gpu = cpu.to(torch::kCUDA);
    gpu.set_requires_grad(requires_grad);
    return gpu;
}

auto torch_to_host(const torch::Tensor& tensor) -> std::vector<float>
{
    auto cpu = tensor.detach().contiguous().to(torch::kCPU);
    const float* begin = cpu.data_ptr<float>();
    return { begin, begin + cpu.numel() };
}

auto shape_to_int64(const std::vector<int>& shape) -> std::vector<int64_t>
{
    std::vector<int64_t> sizes;
    sizes.reserve(shape.size());
    for (int dimension : shape)
    {
        sizes.push_back(static_cast<int64_t>(dimension));
    }
    return sizes;
}

} // namespace

class TorchEquivalenceTest : public GpuTest
{
protected:
    void SetUp() override
    {
        GpuTest::SetUp();
        if (!torch::cuda::is_available())
        {
            GTEST_SKIP() << "LibTorch CUDA is not available";
        }
    }
};

TEST_F(TorchEquivalenceTest, Conv2dForwardAndBackwardMatchLibTorch)
{
    // Given: Identical NCHW input, NCHW weights, and bias on dl and LibTorch
    const std::vector<int> input_shape = { 2, 3, 8, 8 };
    const std::vector<int> weight_shape = { 4, 3, 3, 3 };
    const std::vector<float> input_host = random_host(2 * 3 * 8 * 8, 11U);
    const std::vector<float> weight_host = random_host(4 * 3 * 3 * 3, 22U);
    const std::vector<float> bias_host = random_host(4, 33U);
    Tensor dl_input = Tensor::from_host(input_shape, input_host, Device::GPU);
    Conv2d conv(3, 4, 3, 1, 1, 0.0F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", weight_shape, weight_host);
    set_named_parameter(params, "bias", { 1, 4, 1, 1 }, bias_host);
    conv.set_parameters(params);

    auto torch_input = torch_from_host(shape_to_int64(input_shape), input_host, true);
    auto torch_weight = torch_from_host({ 4, 3, 3, 3 }, weight_host, true);
    auto torch_bias = torch_from_host({ 4 }, bias_host, true);

    // When: Forward and backward are run in both frameworks with the same upstream gradient
    Tensor dl_output = conv.forward(dl_input);
    synchronize_device();
    const std::vector<float> output_host = dl_output.to_host();
    auto torch_output = torch::conv2d(torch_input, torch_weight, torch_bias, /*stride=*/1, /*padding=*/1);
    const std::vector<float> grad_host = random_host(output_host.size(), 44U);
    Tensor dl_grad_output = Tensor::from_host(dl_output.get_shape(), grad_host, Device::GPU);
    Tensor dl_grad_input = conv.backward(dl_grad_output);
    synchronize_device();
    auto torch_grad = torch_from_host(shape_to_int64(dl_output.get_shape()), grad_host, false);
    torch_output.backward(torch_grad);

    // Then: Activations and input gradients match within 1e-4
    expect_near_vector(output_host, torch_to_host(torch_output), kTorchTol);
    expect_near_vector(dl_grad_input.to_host(), torch_to_host(torch_input.grad()), kTorchTol);
}

TEST_F(TorchEquivalenceTest, MaxPool2dForwardAndBackwardMatchLibTorch)
{
    // Given: Identical NCHW inputs with unique values so argmax routing is unambiguous
    const std::vector<int> input_shape = { 2, 2, 8, 8 };
    const std::vector<float> input_host = unique_host(input_shape, 55U);
    Tensor dl_input = Tensor::from_host(input_shape, input_host, Device::GPU);
    MaxPool2d pool(2, 2);
    auto torch_input = torch_from_host(shape_to_int64(input_shape), input_host, true);

    // When: 2x2 stride-2 max pooling is applied and a random upstream gradient is backpropagated
    Tensor dl_output = pool.forward(dl_input);
    synchronize_device();
    auto torch_output = torch::max_pool2d(torch_input, { 2, 2 }, { 2, 2 });
    const std::vector<float> grad_host = random_host(dl_output.get_size(), 66U);
    Tensor dl_grad_output = Tensor::from_host(dl_output.get_shape(), grad_host, Device::GPU);
    Tensor dl_grad_input = pool.backward(dl_grad_output);
    synchronize_device();
    torch_output.backward(torch_from_host(shape_to_int64(dl_output.get_shape()), grad_host, false));

    // Then: Pooled activations and routed gradients match within 1e-4
    expect_near_vector(dl_output.to_host(), torch_to_host(torch_output), kTorchTol);
    expect_near_vector(dl_grad_input.to_host(), torch_to_host(torch_input.grad()), kTorchTol);
}

TEST_F(TorchEquivalenceTest, FullyConnectedForwardAndBackwardMatchLibTorch)
{
    // Given: Identical rank-2 inputs; LibTorch Linear stores W as [out, in], dl stores [in, out]
    const int batch = 4;
    const int in_features = 6;
    const int out_features = 5;
    const std::vector<float> input_host = random_host(static_cast<std::size_t>(batch * in_features), 77U);
    const std::vector<float> weight_host = random_host(static_cast<std::size_t>(in_features * out_features), 88U);
    const std::vector<float> bias_host = random_host(static_cast<std::size_t>(out_features), 99U);
    Tensor dl_input = Tensor::from_host({ batch, in_features }, input_host, Device::GPU);
    FullyConnected dense(in_features, out_features, 0.0F);
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", { in_features, out_features }, weight_host);
    set_named_parameter(params, "bias", { 1, out_features }, bias_host);
    dense.set_parameters(params);

    auto torch_input = torch_from_host({ batch, in_features }, input_host, true);
    auto torch_weight_io = torch_from_host({ in_features, out_features }, weight_host, false);
    auto torch_weight = torch_weight_io.transpose(0, 1).contiguous();
    torch_weight.set_requires_grad(true);
    auto torch_bias = torch_from_host({ out_features }, bias_host, true);

    // When: Y = XW + b is evaluated and the same upstream gradient is backpropagated
    Tensor dl_output = dense.forward(dl_input);
    synchronize_device();
    auto torch_output = torch::linear(torch_input, torch_weight, torch_bias);
    const std::vector<float> grad_host = random_host(static_cast<std::size_t>(batch * out_features), 111U);
    Tensor dl_grad_output = Tensor::from_host({ batch, out_features }, grad_host, Device::GPU);
    Tensor dl_grad_input = dense.backward(dl_grad_output);
    synchronize_device();
    torch_output.backward(torch_from_host({ batch, out_features }, grad_host, false));

    // Then: Activations and dX match within 1e-4
    expect_near_vector(dl_output.to_host(), torch_to_host(torch_output), kTorchTol);
    expect_near_vector(dl_grad_input.to_host(), torch_to_host(torch_input.grad()), kTorchTol);
}
