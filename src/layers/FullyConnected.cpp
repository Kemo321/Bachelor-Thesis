#include "DeepLearnLib/layers/FullyConnected.hpp"
#include "DeepLearnLib/utils/ExperimentSeed.hpp"
#include "DeepLearnLib/utils/Nvtx.hpp"

#include <cublasLt.h>

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int kFillThreads = 256;

__global__ void fill_constant_kernel(float* out, int count, float value)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
    {
        out[index] = value;
    }
}

auto fill_uniform(dl::Tensor& tensor, float low, float high, unsigned long long seed) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    const int count = static_cast<int>(tensor.get_size());
    std::vector<float> host(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index)
    {
        host[static_cast<std::size_t>(index)] = dl::paired_uniform(index, low, high, seed);
    }
    CHECK_CUDA(cudaMemcpyAsync(tensor.data(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice,
        dl::current_stream()));
    CHECK_CUDA(cudaStreamSynchronize(dl::current_stream()));
}

auto fill_constant(dl::Tensor& tensor, float value) -> void
{
    if (tensor.get_size() == 0)
    {
        return;
    }
    if (value == 0.0F)
    {
        CHECK_CUDA(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), dl::current_stream()));
        CHECK_CUDA(cudaGetLastError());
        return;
    }
    const int count = static_cast<int>(tensor.get_size());
    const dim3 grid(static_cast<unsigned int>((count + kFillThreads - 1) / kFillThreads));
    fill_constant_kernel<<<grid, kFillThreads, 0, dl::current_stream()>>>(tensor.data(), count, value);
    CHECK_CUDA(cudaGetLastError());
}

auto ensure_zero_like(std::optional<dl::Tensor>& slot, const dl::Tensor& like) -> dl::Tensor&
{
    if (!slot.has_value() || slot->get_shape() != like.get_shape() || slot->get_dtype() != like.get_dtype())
    {
        slot = dl::Tensor(like.get_shape(), like.get_device(), like.get_dtype());
        fill_constant(*slot, 0.0F);
    }
    return *slot;
}

auto require_gpu(const dl::Tensor& tensor, const char* name) -> void
{
    if (tensor.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " must reside on the GPU");
    }
    if (tensor.get_size() > 0 && tensor.data() == nullptr)
    {
        throw std::runtime_error(std::string(name) + " has a null device pointer");
    }
}

auto copy_same_size(dl::Tensor& dst, const dl::Tensor& src, const char* name) -> void
{
    if (src.get_device() != dl::Device::GPU || dst.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " requires GPU tensors");
    }
    if (src.get_size() != dst.get_size())
    {
        throw std::runtime_error(std::string(name) + " tensor size mismatch");
    }
    if (src.get_size() == 0)
    {
        return;
    }
    if (src.get_dtype() == dst.get_dtype())
    {
        dl::memcpy_d2d_on_current(dst.data(), src.data(), src.nbytes());
        return;
    }
    const dl::Tensor converted = src.to_dtype(dst.get_dtype(), dl::current_stream());
    dl::memcpy_d2d_on_current(dst.data(), converted.data(), dst.nbytes());
}

constexpr int kSmallGemmMacs = 524288;

__global__ void gemm_bias_small_kernel(const float* input, const float* weight, const float* bias, float* output,
    int batch, int inputs, int outputs)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int total = batch * outputs;
    if (index >= total)
    {
        return;
    }
    const int sample = index / outputs;
    const int column = index - (sample * outputs);
    const float* row = input + (static_cast<long long>(sample) * inputs);
    float sum = bias[column];
    for (int inner = 0; inner < inputs; ++inner)
    {
        sum += row[inner] * weight[(static_cast<long long>(inner) * outputs) + column];
    }
    output[index] = sum;
}

auto require_rank2(const dl::Tensor& tensor, int expected_cols, const char* name) -> void
{
    require_gpu(tensor, name);
    if (tensor.get_shape().size() != 2)
    {
        throw std::runtime_error(std::string(name) + " must be rank-2 [batch, features], got "
            + tensor.describe());
    }
    if (tensor.get_shape()[1] != expected_cols)
    {
        throw std::runtime_error(std::string(name) + " has an unexpected feature dimension (expected "
            + std::to_string(expected_cols) + ", got " + tensor.describe() + ")");
    }
}

auto fullyconnected_weight_shape(int input_size, int output_size) -> std::vector<int>
{
    if (input_size <= 0 || output_size <= 0)
    {
        throw std::runtime_error("FullyConnected requires positive input and output sizes");
    }
    return { input_size, output_size };
}

}

namespace
{

constexpr std::size_t kGemmBiasWorkspaceBytes = 4ULL * 1024ULL * 1024ULL;

auto destroy_layout(cublasLtMatrixLayout_t& layout) -> void
{
    if (layout != nullptr)
    {
        cublasLtMatrixLayoutDestroy(layout);
        layout = nullptr;
    }
}

}

struct FullyConnectedGemmBias
{
    cublasLtHandle_t handle { nullptr };
    cublasLtMatmulDesc_t operation { nullptr };
    cublasLtMatrixLayout_t weights_layout { nullptr };
    cublasLtMatrixLayout_t input_layout { nullptr };
    cublasLtMatrixLayout_t output_layout { nullptr };
    cublasLtMatmulPreference_t preference { nullptr };
    cublasLtMatmulAlgo_t algorithm {};
    void* workspace { nullptr };
    int batch { -1 };
    int inputs { -1 };
    int outputs { -1 };
    bool ready { false };
    bool rejected { false };

    FullyConnectedGemmBias()
    {
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        {
            handle = nullptr;
            return;
        }
        if (cudaMalloc(&workspace, kGemmBiasWorkspaceBytes) != cudaSuccess)
        {
            workspace = nullptr;
            return;
        }
        cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);
        const cublasOperation_t identity = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &identity, sizeof(identity));
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &identity, sizeof(identity));
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        cublasLtMatmulPreferenceCreate(&preference);
        const std::size_t bytes = kGemmBiasWorkspaceBytes;
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes));
    }

    ~FullyConnectedGemmBias()
    {
        destroy_layout(weights_layout);
        destroy_layout(input_layout);
        destroy_layout(output_layout);
        if (preference != nullptr)
        {
            cublasLtMatmulPreferenceDestroy(preference);
        }
        if (operation != nullptr)
        {
            cublasLtMatmulDescDestroy(operation);
        }
        if (workspace != nullptr)
        {
            cudaFree(workspace);
        }
        if (handle != nullptr)
        {
            cublasLtDestroy(handle);
        }
    }

    FullyConnectedGemmBias(const FullyConnectedGemmBias&) = delete;
    auto operator=(const FullyConnectedGemmBias&) -> FullyConnectedGemmBias& = delete;

    auto prepare(int next_batch, int next_inputs, int next_outputs, const void* bias) -> bool
    {
        if (handle == nullptr || operation == nullptr || workspace == nullptr || next_batch < 1)
        {
            return false;
        }
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        if (ready && batch == next_batch && inputs == next_inputs && outputs == next_outputs)
        {
            return true;
        }
        if (rejected && batch == next_batch && inputs == next_inputs && outputs == next_outputs)
        {
            return false;
        }

        destroy_layout(weights_layout);
        destroy_layout(input_layout);
        destroy_layout(output_layout);
        ready = false;
        batch = next_batch;
        inputs = next_inputs;
        outputs = next_outputs;

        if (cublasLtMatrixLayoutCreate(&weights_layout, CUDA_R_32F, next_outputs, next_inputs, next_outputs)
                != CUBLAS_STATUS_SUCCESS
            || cublasLtMatrixLayoutCreate(&input_layout, CUDA_R_32F, next_inputs, next_batch, next_inputs)
                != CUBLAS_STATUS_SUCCESS
            || cublasLtMatrixLayoutCreate(&output_layout, CUDA_R_32F, next_outputs, next_batch, next_outputs)
                != CUBLAS_STATUS_SUCCESS)
        {
            rejected = true;
            return false;
        }

        cublasLtMatmulHeuristicResult_t heuristic {};
        int returned = 0;
        const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, operation, weights_layout, input_layout,
            output_layout, output_layout, preference, 1, &heuristic, &returned);
        if (status != CUBLAS_STATUS_SUCCESS || returned < 1 || heuristic.state != CUBLAS_STATUS_SUCCESS)
        {
            rejected = true;
            return false;
        }
        algorithm = heuristic.algo;
        rejected = false;
        ready = true;
        return true;
    }

    auto reject() -> void
    {
        ready = false;
        rejected = true;
    }

    auto run(const void* weights, const void* input, void* output, cudaStream_t stream) const -> bool
    {
        const float alpha = 1.0F;
        const float beta = 0.0F;
        return cublasLtMatmul(handle, operation, &alpha, weights, weights_layout, input, input_layout, &beta, output,
                   output_layout, output, output_layout, &algorithm, workspace, kGemmBiasWorkspaceBytes, stream)
            == CUBLAS_STATUS_SUCCESS;
    }
};

FullyConnected::FullyConnected(int input_size, int output_size, float inertia_val)
    : weights_(fullyconnected_weight_shape(input_size, output_size), dl::Device::GPU)
    , biases_({ 1, output_size }, dl::Device::GPU)
    , weights_gradient_({ input_size, output_size }, dl::Device::GPU)
    , biases_gradient_({ 1, output_size }, dl::Device::GPU)
    , input_size_(input_size)
    , output_size_(output_size)
    , inertia_(inertia_val)
{
    device_ = dl::Device::GPU;
    const float bound = dl::uniform_bound(static_cast<float>(input_size_));
    fill_uniform(weights_, -bound, bound, dl::kLinearWeightSeed);
    fill_uniform(biases_, -bound, bound, dl::kLinearBiasSeed);
    fill_constant(weights_gradient_, 0.0F);
    fill_constant(biases_gradient_, 0.0F);
}

FullyConnected::~FullyConnected() = default;

auto FullyConnected::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("FullyConnected_Forward");
    const dl::StreamGuard stream_guard(stream);
    require_rank2(input_tensor, input_size_, "FullyConnected::forward input");

    if (input_tensor.get_dtype() != weights_.get_dtype())
    {
        input_cache_ = input_tensor.to_dtype(weights_.get_dtype(), stream);
    }
    else
    {
        input_cache_ = input_tensor.as_view();
    }
    input_cache_ready_ = true;

    dl::Tensor& output = dl::Tensor::ensure(output_cache_, { input_cache_->get_shape()[0], output_size_ },
        dl::Device::GPU, weights_.get_dtype());
    const int batch = input_cache_->get_shape()[0];
    if (weights_.get_dtype() == dl::Dtype::Float32 && batch > 0
        && static_cast<long long>(batch) * input_size_ * output_size_ <= kSmallGemmMacs)
    {
        const int total = batch * output_size_;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        gemm_bias_small_kernel<<<blocks, threads, 0, stream>>>(input_cache_->data(), weights_.data(), biases_.data(),
            output.data(), batch, input_size_, output_size_);
        CHECK_CUDA(cudaGetLastError());
        return output.as_view();
    }
    if (weights_.get_dtype() == dl::Dtype::Float32)
    {
        if (gemm_bias_ == nullptr)
        {
            gemm_bias_ = std::make_unique<FullyConnectedGemmBias>();
        }
        if (gemm_bias_->prepare(batch, input_size_, output_size_, biases_.data()))
        {
            if (gemm_bias_->run(weights_.data(), input_cache_->data(), output.data(), stream))
            {
                return output.as_view();
            }
            gemm_bias_->reject();
        }
    }
    input_cache_->matmul_into(weights_, output);
    output.add_row_(biases_);
    return output.as_view();
}

auto FullyConnected::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("FullyConnected_Backward");
    const dl::StreamGuard stream_guard(stream);
    if (!input_cache_ready_ || !input_cache_.has_value())
    {
        throw std::runtime_error("FullyConnected::backward requires a preceding forward pass");
    }
    require_rank2(output_error_derivative, output_size_, "FullyConnected::backward grad_output");
    if (output_error_derivative.get_shape()[0] != input_cache_->get_shape()[0])
    {
        throw std::runtime_error("FullyConnected::backward batch size does not match the cached input");
    }

    const dl::Tensor* grad_output = &output_error_derivative;
    dl::Tensor converted_grad;
    if (output_error_derivative.get_dtype() != weights_.get_dtype())
    {
        converted_grad = output_error_derivative.to_dtype(weights_.get_dtype(), stream);
        grad_output = &converted_grad;
    }

    input_cache_->matmul_into(*grad_output, weights_gradient_, true, false, inertia_);
    biases_gradient_.add_sum_rows_(*grad_output, inertia_);

    dl::Tensor& grad_input = dl::Tensor::ensure(grad_input_cache_, input_cache_->get_shape(), dl::Device::GPU,
        weights_.get_dtype());

    grad_output->matmul_into(weights_, grad_input, false, true, 0.0F);
    input_cache_ready_ = false;
    return grad_input.as_view();
}

void FullyConnected::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("FullyConnected_Step");
    const dl::StreamGuard stream_guard(stream);
    if (frozen())
    {
        return;
    }
    const float clip = parameter_clip_bound();
    const float lr = step_learning_rate();
    if (use_adam)
    {
        ++adam_step;
        dl::Tensor& weight_moment = ensure_zero_like(weights_velocity_, weights_);
        dl::Tensor& weight_second = ensure_zero_like(weights_second_, weights_);
        dl::Tensor& bias_moment = ensure_zero_like(biases_velocity_, biases_);
        dl::Tensor& bias_second = ensure_zero_like(biases_second_, biases_);
        weights_.adam_update_(weights_gradient_, weight_moment, weight_second, lr, 0.9F, 0.999F, weight_decay, 1.0e-8F,
            adam_step, clip);
        biases_.adam_update_(biases_gradient_, bias_moment, bias_second, lr, 0.9F, 0.999F, weight_decay, 1.0e-8F,
            adam_step, clip);
        return;
    }
    if (momentum > 0.0F)
    {
        dl::Tensor& weight_velocity = ensure_zero_like(weights_velocity_, weights_);
        dl::Tensor& bias_velocity = ensure_zero_like(biases_velocity_, biases_);
        weights_.sgd_momentum_update_(weights_gradient_, weight_velocity, lr, momentum, weight_decay, clip);
        biases_.sgd_momentum_update_(biases_gradient_, bias_velocity, lr, momentum, weight_decay, clip);
        return;
    }
    weights_.sgd_update_(weights_gradient_, lr, weight_decay, clip);
    biases_.sgd_update_(biases_gradient_, lr, weight_decay, clip);
}

void FullyConnected::clip_gradients(float abs_bound, cudaStream_t stream)
{
    const dl::StreamGuard stream_guard(stream);
    if (abs_bound <= 0.0F)
    {
        return;
    }
    weights_gradient_.clamp_(-abs_bound, abs_bound);
    biases_gradient_.clamp_(-abs_bound, abs_bound);
}

auto FullyConnected::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params;
    params.emplace("weights", weights_.view(weights_.get_shape()));
    params.emplace("bias", biases_.view(biases_.get_shape()));
    return params;
}

void FullyConnected::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    copy_same_size(weights_, params.at("weights"), "FullyConnected::set_parameters weights");
    copy_same_size(biases_, params.at("bias"), "FullyConnected::set_parameters bias");
}

auto FullyConnected::to(dl::Device device) -> void
{
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("FullyConnected parameters must remain on the GPU");
    }
    device_ = device;
}
