#include "DeepLearnLib/Losses.hpp"
#include "DeepLearnLib/SafeMath.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int kSoftmaxThreads = 256;
constexpr int kLossThreads = 256;
constexpr int kReduceThreads = 256;

auto require_same_gpu(const dl::Tensor& target, const dl::Tensor& prediction, const char* name) -> void
{
    if (target.get_device() != dl::Device::GPU || prediction.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " requires GPU tensors");
    }
    if (target.get_shape() != prediction.get_shape())
    {
        throw std::runtime_error(std::string(name) + " requires identically shaped tensors (target "
            + target.describe() + " vs prediction " + prediction.describe() + ")");
    }
    if (target.get_size() != prediction.get_size())
    {
        throw std::runtime_error(std::string(name) + " requires identically sized tensors (target "
            + target.describe() + " vs prediction " + prediction.describe() + ")");
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
        throw std::runtime_error(std::string(name) + " requires rank-2 [batch, classes] tensors, got "
            + tensor.describe());
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

    // Max-subtraction: exp(x - max(x)) stays in (0, 1] and avoids overflow.
    float row_max = input_row[0];
    for (int col = 1; col < classes; ++col)
    {
        row_max = fmaxf(row_max, input_row[col]);
    }
    if (!isfinite(row_max))
    {
        const float uniform = dl::safe_inv(static_cast<float>(classes));
        for (int col = 0; col < classes; ++col)
        {
            output_row[col] = uniform;
        }
        return;
    }

    float sum = 0.0F;
    for (int col = 0; col < classes; ++col)
    {
        const float exp_value = expf(input_row[col] - row_max);
        output_row[col] = exp_value;
        sum += exp_value;
    }
    sum = fmaxf(sum, dl::kSafeEps);
    for (int col = 0; col < classes; ++col)
    {
        output_row[col] = output_row[col] / sum;
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
        const float probability = dl::clamp_unit(probabilities[offset + static_cast<std::size_t>(col)]);
        loss -= target[offset + static_cast<std::size_t>(col)] * logf(probability);
    }
    row_loss[row] = loss;
}

__global__ void mse_sqdiff_sum_kernel(const float* prediction, const float* target, float* out, int count)
{
    __shared__ float shared_sum[kReduceThreads];
    float partial = 0.0F;
    for (int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x); index < count;
         index += static_cast<int>(blockDim.x * gridDim.x))
    {
        const float delta = prediction[index] - target[index];
        partial += delta * delta;
    }
    shared_sum[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = kReduceThreads / 2; stride > 0; stride >>= 1)
    {
        if (static_cast<int>(threadIdx.x) < stride)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(out, shared_sum[0]);
    }
}

__global__ void mse_grad_kernel(const float* prediction, const float* target, float* gradient, int count, float scale)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
    {
        gradient[index] = (prediction[index] - target[index]) * scale;
    }
}

__global__ void softmax_minus_target_kernel(
    const float* probabilities, const float* target, float* gradient, int count, float inv_batch)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
    {
        gradient[index] = (probabilities[index] - target[index]) * inv_batch;
    }
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

    dl::Tensor sum_squares({ 1 }, dl::Device::GPU);
    CHECK_CUDA(cudaMemsetAsync(sum_squares.data(), 0, sizeof(float), dl::current_stream()));
    const int count = static_cast<int>(prediction.get_size());
    const int blocks = std::max(1, (count + kReduceThreads - 1) / kReduceThreads);
    mse_sqdiff_sum_kernel<<<static_cast<unsigned int>(std::min(blocks, 1024)), kReduceThreads, 0, dl::current_stream()>>>(
        prediction.data(), target.data(), sum_squares.data(), count);
    CHECK_CUDA(cudaGetLastError());
    const float mean = dl::safe_div(sum_squares.to_host()[0], static_cast<float>(prediction.get_size()));
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

    const float scale = 2.0F * dl::safe_inv(static_cast<float>(prediction.get_size()));
    const int count = static_cast<int>(prediction.get_size());
    const dim3 grid(static_cast<unsigned int>((count + kLossThreads - 1) / kLossThreads));
    mse_grad_kernel<<<grid, kLossThreads, 0, dl::current_stream()>>>(
        prediction.data(), target.data(), gradient.data(), count, scale);
    CHECK_CUDA(cudaGetLastError());
    return gradient;
}

auto CrossEntropyLoss::loss(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor
{
    require_same_gpu(target, prediction, "CrossEntropyLoss::loss");
    require_rank2(prediction, "CrossEntropyLoss::loss");

    const dl::Tensor pred_f32 = prediction.to_dtype(dl::Dtype::Float32);
    const dl::Tensor tgt_f32 = target.to_dtype(dl::Dtype::Float32);

    const int batch = pred_f32.get_shape()[0];
    const int classes = pred_f32.get_shape()[1];
    dl::Tensor probabilities = softmax_probabilities(pred_f32);
    dl::Tensor row_loss({ batch }, dl::Device::GPU);
    const int blocks = (batch + kSoftmaxThreads - 1) / kSoftmaxThreads;
    cross_entropy_rows_kernel<<<blocks, kSoftmaxThreads>>>(probabilities.data(), tgt_f32.data(), row_loss.data(), batch,
        classes);
    CHECK_CUDA(cudaGetLastError());

    const float total = row_loss.sum().to_host()[0];
    return scalar_from_host(dl::safe_div(total, static_cast<float>(batch)));
}

auto CrossEntropyLoss::loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction) -> dl::Tensor
{
    require_same_gpu(target, prediction, "CrossEntropyLoss::loss_derivative");
    require_rank2(prediction, "CrossEntropyLoss::loss_derivative");

    const dl::Dtype result_dtype = prediction.get_dtype();
    const dl::Tensor pred_f32 = prediction.to_dtype(dl::Dtype::Float32);
    const dl::Tensor tgt_f32 = target.to_dtype(dl::Dtype::Float32);

    const int batch = pred_f32.get_shape()[0];
    dl::Tensor probabilities = softmax_probabilities(pred_f32);
    dl::Tensor gradient(pred_f32.get_shape(), dl::Device::GPU);
    const float inv_batch = dl::safe_inv(static_cast<float>(batch));
    const int count = static_cast<int>(pred_f32.get_size());
    const dim3 grid(static_cast<unsigned int>((count + kLossThreads - 1) / kLossThreads));
    softmax_minus_target_kernel<<<grid, kLossThreads, 0, dl::current_stream()>>>(
        probabilities.data(), tgt_f32.data(), gradient.data(), count, inv_batch);
    CHECK_CUDA(cudaGetLastError());
    gradient = gradient * dl::loss_scale();
    return gradient.to_dtype(result_dtype);
}
