#include "test_helpers.hpp"

#include "DeepLearnLib/BatchNorm2d.hpp"
#include "DeepLearnLib/Conv2d.hpp"
#include "DeepLearnLib/FullyConnected.hpp"
#include "DeepLearnLib/FusedCBR2d.hpp"
#include "DeepLearnLib/MaxPool2d.hpp"
#include "DeepLearnLib/Tensor.hpp"
#include "DeepLearnLib/YOLOLoss.hpp"
#include "TorchYOLO.hpp"

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

constexpr float kTorchTol = 1e-3F;

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

namespace
{

auto channel_vector(int channels, float value) -> std::vector<float>
{
    return std::vector<float>(static_cast<std::size_t>(channels), value);
}

auto torch_yolo_loss(torch::Tensor prediction, torch::Tensor target, bool detach_assignment) -> torch::Tensor
{
    constexpr int kGridSize = 7;
    constexpr float kLambdaCoord = 5.0F;
    constexpr float kLambdaNoobj = 0.5F;
    constexpr float kEps = 1.0e-7F;

    auto as_grid = [](torch::Tensor tensor) -> torch::Tensor
    {
        if (tensor.dim() == 4)
        {
            return tensor;
        }
        const auto batch = tensor.size(0);
        const auto cells = static_cast<int64_t>(kGridSize) * static_cast<int64_t>(kGridSize);
        return tensor.view({ batch, kGridSize, kGridSize, tensor.size(1) / cells });
    };

    torch::Tensor pred = as_grid(prediction).to(torch::kFloat32);
    torch::Tensor tgt = as_grid(target).to(torch::kFloat32);
    const auto batch_size = pred.size(0);
    const auto options = pred.options();
    const auto col = torch::arange(kGridSize, options).view({ 1, 1, kGridSize });
    const auto row = torch::arange(kGridSize, options).view({ 1, kGridSize, 1 });

    const auto p1x = pred.select(-1, 0);
    const auto p1y = pred.select(-1, 1);
    const auto p1w = pred.select(-1, 2);
    const auto p1h = pred.select(-1, 3);
    const auto p1c = pred.select(-1, 4);
    const auto p2x = pred.select(-1, 5);
    const auto p2y = pred.select(-1, 6);
    const auto p2w = pred.select(-1, 7);
    const auto p2h = pred.select(-1, 8);
    const auto p2c = pred.select(-1, 9);
    const auto tx = tgt.select(-1, 0);
    const auto ty = tgt.select(-1, 1);
    const auto tw = tgt.select(-1, 2);
    const auto th = tgt.select(-1, 3);
    const auto obj = tgt.select(-1, 4);

    auto box_iou = [&](const torch::Tensor& cx1, const torch::Tensor& cy1, const torch::Tensor& w1,
                       const torch::Tensor& h1, const torch::Tensor& cx2, const torch::Tensor& cy2,
                       const torch::Tensor& w2, const torch::Tensor& h2)
    {
        const auto b1_x1 = cx1 - (w1 * 0.5);
        const auto b1_y1 = cy1 - (h1 * 0.5);
        const auto b1_x2 = cx1 + (w1 * 0.5);
        const auto b1_y2 = cy1 + (h1 * 0.5);
        const auto b2_x1 = cx2 - (w2 * 0.5);
        const auto b2_y1 = cy2 - (h2 * 0.5);
        const auto b2_x2 = cx2 + (w2 * 0.5);
        const auto b2_y2 = cy2 + (h2 * 0.5);
        const auto inter_w = (torch::min(b1_x2, b2_x2) - torch::max(b1_x1, b2_x1)).clamp_min(0.0);
        const auto inter_h = (torch::min(b1_y2, b2_y2) - torch::max(b1_y1, b2_y1)).clamp_min(0.0);
        const auto inter = inter_w * inter_h;
        const auto area1 = (w1 * h1).clamp_min(kEps);
        const auto area2 = (w2 * h2).clamp_min(kEps);
        return inter / (area1 + area2 - inter + kEps);
    };

    const auto grid = static_cast<float>(kGridSize);
    auto iou1 = box_iou((p1x + col) / grid, (p1y + row) / grid, p1w, p1h, (tx + col) / grid, (ty + row) / grid, tw, th);
    auto iou2 = box_iou((p2x + col) / grid, (p2y + row) / grid, p2w, p2h, (tx + col) / grid, (ty + row) / grid, tw, th);
    if (detach_assignment)
    {
        iou1 = iou1.detach();
        iou2 = iou2.detach();
    }
    auto box2_better = (iou2 > iou1).to(pred.dtype());
    auto resp_b1 = (1.0 - box2_better) * obj;
    auto resp_b2 = box2_better * obj;
    auto noobj_b1 = 1.0 - resp_b1;
    auto noobj_b2 = 1.0 - resp_b2;
    if (detach_assignment)
    {
        resp_b1 = resp_b1.detach();
        resp_b2 = resp_b2.detach();
        noobj_b1 = noobj_b1.detach();
        noobj_b2 = noobj_b2.detach();
    }

    auto sqrt_safe = [](const torch::Tensor& value)
    { return torch::sqrt(value.clamp_min(kEps)); };
    const auto xy_b1 = (p1x - tx).square() + (p1y - ty).square();
    const auto xy_b2 = (p2x - tx).square() + (p2y - ty).square();
    const auto wh_b1 = (sqrt_safe(p1w) - sqrt_safe(tw)).square() + (sqrt_safe(p1h) - sqrt_safe(th)).square();
    const auto wh_b2 = (sqrt_safe(p2w) - sqrt_safe(tw)).square() + (sqrt_safe(p2h) - sqrt_safe(th)).square();
    const auto l_coord = kLambdaCoord * ((xy_b1 * resp_b1) + (xy_b2 * resp_b2) + (wh_b1 * resp_b1) + (wh_b2 * resp_b2));
    const auto conf_obj = ((p1c - iou1).square() * resp_b1) + ((p2c - iou2).square() * resp_b2);
    const auto conf_noobj = kLambdaNoobj * ((p1c.square() * noobj_b1) + (p2c.square() * noobj_b2));
    const auto class_err = (pred.slice(-1, 10) - tgt.slice(-1, 10)).square().sum(-1) * obj;
    return (l_coord + conf_obj + conf_noobj + class_err).sum() / static_cast<double>(batch_size);
}

} // namespace

TEST_F(TorchEquivalenceTest, BatchNorm2dTrainForwardAndBackwardMatchLibTorch)
{
    const int channels = 4;
    const std::vector<int> input_shape = { 2, channels, 6, 6 };
    const std::vector<float> input_host = random_host(2 * channels * 6 * 6, 121U);
    Tensor dl_input = Tensor::from_host(input_shape, input_host, Device::GPU);
    BatchNorm2d batch_norm(channels);
    batch_norm.train();
    batch_norm.weight_decay = 0.0F;

    const std::vector<float> gamma_host = channel_vector(channels, 1.0F);
    const std::vector<float> beta_host = channel_vector(channels, 0.0F);
    const std::vector<float> mean_host = channel_vector(channels, 0.0F);
    const std::vector<float> var_host = channel_vector(channels, 1.0F);
    auto torch_input = torch_from_host(shape_to_int64(input_shape), input_host, true);
    auto torch_gamma = torch_from_host({ channels }, gamma_host, true);
    auto torch_beta = torch_from_host({ channels }, beta_host, true);
    auto torch_mean = torch_from_host({ channels }, mean_host, false);
    auto torch_var = torch_from_host({ channels }, var_host, false);

    Tensor dl_output = batch_norm.forward(dl_input);
    synchronize_device();
    auto torch_output = torch::batch_norm(
        torch_input, torch_gamma, torch_beta, torch_mean, torch_var, true, 0.1, 1.0e-5, true);
    const std::vector<float> grad_host = random_host(dl_output.get_size(), 131U);
    Tensor dl_grad_input = batch_norm.backward(Tensor::from_host(dl_output.get_shape(), grad_host, Device::GPU));
    synchronize_device();
    torch_output.backward(torch_from_host(shape_to_int64(dl_output.get_shape()), grad_host, false));

    expect_near_vector(dl_output.to_host(), torch_to_host(torch_output), kTorchTol);
    expect_near_vector(dl_grad_input.to_host(), torch_to_host(torch_input.grad()), kTorchTol);
}

TEST_F(TorchEquivalenceTest, FusedCBR2dEvalForwardMatchesLibTorch)
{
    const std::vector<int> input_shape = { 2, 3, 8, 8 };
    const std::vector<int> weight_shape = { 4, 3, 3, 3 };
    const std::vector<float> input_host = random_host(2 * 3 * 8 * 8, 141U);
    const std::vector<float> weight_host = random_host(4 * 3 * 3 * 3, 151U);
    const std::vector<float> bias_host = random_host(4, 161U);
    Tensor dl_input = Tensor::from_host(input_shape, input_host, Device::GPU);
    FusedCBR2d fused(3, 4, 3, 1, 1, 0.1F);
    fused.eval();
    std::map<std::string, Tensor> params;
    set_named_parameter(params, "weights", weight_shape, weight_host);
    set_named_parameter(params, "bias", { 1, 4, 1, 1 }, bias_host);
    set_named_parameter(params, "gamma", { 1, 4, 1, 1 }, channel_vector(4, 1.0F));
    set_named_parameter(params, "beta", { 1, 4, 1, 1 }, channel_vector(4, 0.0F));
    set_named_parameter(params, "running_mean", { 1, 4, 1, 1 }, channel_vector(4, 0.0F));
    set_named_parameter(params, "running_var", { 1, 4, 1, 1 }, channel_vector(4, 1.0F));
    fused.set_parameters(params);

    auto torch_input = torch_from_host(shape_to_int64(input_shape), input_host, false);
    auto torch_weight = torch_from_host({ 4, 3, 3, 3 }, weight_host, false);
    auto torch_bias = torch_from_host({ 4 }, bias_host, false);
    auto torch_gamma = torch_from_host({ 4 }, channel_vector(4, 1.0F), false);
    auto torch_beta = torch_from_host({ 4 }, channel_vector(4, 0.0F), false);
    auto torch_mean = torch_from_host({ 4 }, channel_vector(4, 0.0F), false);
    auto torch_var = torch_from_host({ 4 }, channel_vector(4, 1.0F), false);

    Tensor dl_output = fused.forward(dl_input);
    synchronize_device();
    auto conv = torch::conv2d(torch_input, torch_weight, torch_bias, 1, 1);
    auto normalised = torch::batch_norm(conv, torch_gamma, torch_beta, torch_mean, torch_var, false, 0.1, 1.0e-5, true);
    auto torch_output = torch::leaky_relu(normalised, 0.1);

    expect_near_vector(dl_output.to_host(), torch_to_host(torch_output), kTorchTol);
}

// The scalar matches compute_yolo_loss. The CUDA kernel treats IoU and the
// responsible-box mask as constants, so the gradient oracle detaches those
// terms. Autograd through compute_yolo_loss is not that comparison.
TEST_F(TorchEquivalenceTest, YOLOLossMatchesLibTorchBaseline)
{
    constexpr int kBatch = 2;
    constexpr int kClasses = 2;
    constexpr int kAttributes = 10 + kClasses;
    constexpr int kCells = 7 * 7;
    std::vector<float> pred_host = random_host(static_cast<std::size_t>(kBatch * kCells * kAttributes), 171U, 0.2F);
    std::vector<float> target_host(pred_host.size(), 0.0F);
    for (int batch = 0; batch < kBatch; ++batch)
    {
        for (int cell = 0; cell < kCells; ++cell)
        {
            const std::size_t base
                = static_cast<std::size_t>((batch * kCells) + cell) * static_cast<std::size_t>(kAttributes);
            for (int box = 0; box < 2; ++box)
            {
                const std::size_t box_base = base + static_cast<std::size_t>(box * 5);
                pred_host[box_base + 2] = std::fabs(pred_host[box_base + 2]) + 0.05F;
                pred_host[box_base + 3] = std::fabs(pred_host[box_base + 3]) + 0.05F;
            }
        }
        const std::size_t object_cell = static_cast<std::size_t>(batch * kCells * kAttributes);
        target_host[object_cell + 0] = 0.4F;
        target_host[object_cell + 1] = 0.6F;
        target_host[object_cell + 2] = 0.3F;
        target_host[object_cell + 3] = 0.2F;
        target_host[object_cell + 4] = 1.0F;
        target_host[object_cell + 10] = 1.0F;
    }

    const std::vector<int> shape = { kBatch, 7, 7, kAttributes };
    Tensor dl_pred = Tensor::from_host(shape, pred_host, Device::GPU);
    Tensor dl_target = Tensor::from_host(shape, target_host, Device::GPU);
    auto torch_pred = torch_from_host(shape_to_int64(shape), pred_host, true);
    auto torch_target = torch_from_host(shape_to_int64(shape), target_host, false);

    const float custom_loss = YOLOLoss::loss(dl_target, dl_pred, kClasses).to_host().front();
    const float baseline_loss = compute_yolo_loss(torch_pred.detach(), torch_target).item<float>();
    const float formula_loss = torch_yolo_loss(torch_pred.detach(), torch_target, false).item<float>();
    auto tracked = torch_yolo_loss(torch_pred, torch_target, true);
    tracked.backward();
    Tensor dl_grad = YOLOLoss::loss_derivative(dl_target, dl_pred, kClasses);
    synchronize_device();

    EXPECT_NEAR(custom_loss, baseline_loss, kTorchTol);
    EXPECT_NEAR(formula_loss, baseline_loss, kTorchTol);
    expect_near_vector(dl_grad.to_host(), torch_to_host(torch_pred.grad()), kTorchTol);
}
