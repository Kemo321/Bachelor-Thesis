#include "DeepLearnLib/Losses.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace
{

constexpr float kLogEps = 1e-8F;
constexpr int kSoftmaxThreads = 256;

struct SquaredDiffAt
{
    const float* prediction;
    const float* target;

    __host__ __device__ auto operator()(int index) const -> float
    {
        const float delta = prediction[index] - target[index];
        return delta * delta;
    }
};

struct MeanSquareGrad
{
    float scale;

    __host__ __device__ auto operator()(float prediction, float target) const -> float
    {
        return (prediction - target) * scale;
    }
};

struct SoftmaxMinusTarget
{
    float inv_batch;

    __host__ __device__ auto operator()(float softmax_value, float target) const -> float
    {
        return (softmax_value - target) * inv_batch;
    }
};

auto require_same_gpu(const dl::Tensor& target, const dl::Tensor& prediction, const char* name) -> void
{
    if (target.get_device() != dl::Device::GPU || prediction.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " requires GPU tensors");
    }
    if (target.get_shape() != prediction.get_shape())
    {
        throw std::runtime_error(std::string(name) + " requires identically shaped tensors");
    }
    if (target.get_size() != prediction.get_size())
    {
        throw std::runtime_error(std::string(name) + " requires identically sized tensors");
    }
    if (target.get_size() > 0 && (target.data() == nullptr || prediction.data() == nullptr))
    {
        throw std::runtime_error(std::string(name) + " has a null device pointer");
    }
}

auto require_rank2(const dl::Tensor& tensor, const char* name) -> void
{
    if (tensor.get_shape().size() != 2)
    {
        throw std::runtime_error(std::string(name) + " requires rank-2 [batch, classes] tensors");
    }
    if (tensor.get_shape()[0] <= 0 || tensor.get_shape()[1] <= 0)
    {
        throw std::runtime_error(std::string(name) + " requires positive batch and class counts");
    }
}

auto scalar_from_host(float value) -> dl::Tensor
{
    return dl::Tensor::from_host({ 1 }, std::vector<float> { value }, dl::Device::GPU);
}

__global__ void softmax_rows_kernel(const float* logits, float* probabilities, int batch, int classes)
{
    const int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row >= batch)
    {
        return;
    }
    const float* input_row = logits + (static_cast<std::size_t>(row) * static_cast<std::size_t>(classes));
    float* output_row = probabilities + (static_cast<std::size_t>(row) * static_cast<std::size_t>(classes));

    float row_max = input_row[0];
    for (int col = 1; col < classes; ++col)
    {
        row_max = fmaxf(row_max, input_row[col]);
    }

    float sum = 0.0F;
    for (int col = 0; col < classes; ++col)
    {
        const float exp_value = expf(input_row[col] - row_max);
        output_row[col] = exp_value;
        sum += exp_value;
    }
    sum = fmaxf(sum, kLogEps);
    for (int col = 0; col < classes; ++col)
    {
        output_row[col] /= sum;
    }
}

__global__ void cross_entropy_rows_kernel(const float* probabilities, const float* target, float* row_loss, int batch,
    int classes)
{
    const int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row >= batch)
    {
        return;
    }
    const std::size_t offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(classes);
    float loss = 0.0F;
    for (int col = 0; col < classes; ++col)
    {
        const float probability = fmaxf(probabilities[offset + static_cast<std::size_t>(col)], kLogEps);
        loss -= target[offset + static_cast<std::size_t>(col)] * logf(probability);
    }
    row_loss[row] = loss;
}

auto softmax_probabilities(const dl::Tensor& logits) -> dl::Tensor
{
    const int batch = logits.get_shape()[0];
    const int classes = logits.get_shape()[1];
    dl::Tensor probabilities(logits.get_shape(), dl::Device::GPU);
    const int blocks = (batch + kSoftmaxThreads - 1) / kSoftmaxThreads;
    softmax_rows_kernel<<<blocks, kSoftmaxThreads>>>(logits.data(), probabilities.data(), batch, classes);
    CHECK_CUDA(cudaGetLastError());
    return probabilities;
}

} // namespace

auto MSELoss::loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor
{
    require_same_gpu(target, prediction, "MSELoss::loss");
    if (prediction.get_size() == 0)
    {
        return scalar_from_host(0.0F);
    }

    const float sum_squares = thrust::transform_reduce(thrust::device, thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(prediction.get_size())),
        SquaredDiffAt { prediction.data(), target.data() }, 0.0F, thrust::plus<float>());
    CHECK_CUDA(cudaGetLastError());
    const float mean = sum_squares / static_cast<float>(prediction.get_size());
    return scalar_from_host(mean);
}

auto MSELoss::loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor
{
    require_same_gpu(target, prediction, "MSELoss::loss_derivative");
    dl::Tensor gradient(prediction.get_shape(), dl::Device::GPU);
    if (prediction.get_size() == 0)
    {
        return gradient;
    }

    const float scale = 2.0F / static_cast<float>(prediction.get_size());
    auto pred_ptr = thrust::device_pointer_cast(prediction.data());
    auto tgt_ptr = thrust::device_pointer_cast(target.data());
    auto grad_ptr = thrust::device_pointer_cast(gradient.data());
    thrust::transform(thrust::device, pred_ptr, pred_ptr + static_cast<std::ptrdiff_t>(prediction.get_size()), tgt_ptr,
        grad_ptr, MeanSquareGrad { scale });
    CHECK_CUDA(cudaGetLastError());
    return gradient;
}

auto CrossEntropyLoss::loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor
{
    require_same_gpu(target, prediction, "CrossEntropyLoss::loss");
    require_rank2(prediction, "CrossEntropyLoss::loss");

    const int batch = prediction.get_shape()[0];
    const int classes = prediction.get_shape()[1];
    dl::Tensor probabilities = softmax_probabilities(prediction);
    dl::Tensor row_loss({ batch }, dl::Device::GPU);
    const int blocks = (batch + kSoftmaxThreads - 1) / kSoftmaxThreads;
    cross_entropy_rows_kernel<<<blocks, kSoftmaxThreads>>>(probabilities.data(), target.data(), row_loss.data(), batch,
        classes);
    CHECK_CUDA(cudaGetLastError());

    const float total = row_loss.sum().to_host()[0];
    return scalar_from_host(total / static_cast<float>(batch));
}

auto CrossEntropyLoss::loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor
{
    require_same_gpu(target, prediction, "CrossEntropyLoss::loss_derivative");
    require_rank2(prediction, "CrossEntropyLoss::loss_derivative");

    const int batch = prediction.get_shape()[0];
    dl::Tensor probabilities = softmax_probabilities(prediction);
    dl::Tensor gradient(prediction.get_shape(), dl::Device::GPU);
    auto prob_ptr = thrust::device_pointer_cast(probabilities.data());
    auto tgt_ptr = thrust::device_pointer_cast(target.data());
    auto grad_ptr = thrust::device_pointer_cast(gradient.data());
    const float inv_batch = 1.0F / static_cast<float>(batch);
    thrust::transform(thrust::device, prob_ptr, prob_ptr + static_cast<std::ptrdiff_t>(prediction.get_size()), tgt_ptr,
        grad_ptr, SoftmaxMinusTarget { inv_batch });
    CHECK_CUDA(cudaGetLastError());
    return gradient;
}
