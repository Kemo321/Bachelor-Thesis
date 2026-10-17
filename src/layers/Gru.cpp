#include "DeepLearnLib/layers/Gru.hpp"

#include "DeepLearnLib/layers/Conv2d.hpp"
#include "DeepLearnLib/utils/ExperimentSeed.hpp"
#include "DeepLearnLib/utils/Nvtx.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr int kFillThreads = 256;

__global__ void batch_to_time_kernel(const float* batch_major, float* time_major, int batch, int time, int features)
{
    const int count = batch * time * features;
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count)
    {
        return;
    }
    const int feature = index % features;
    const int step = (index / features) % time;
    const int sample = index / (features * time);
    time_major[(((step * batch) + sample) * features) + feature] = batch_major[index];
}

__global__ void time_to_batch_kernel(const float* time_major, float* batch_major, int batch, int time, int features)
{
    const int count = batch * time * features;
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count)
    {
        return;
    }
    const int feature = index % features;
    const int step = (index / features) % time;
    const int sample = index / (features * time);
    batch_major[index] = time_major[(((step * batch) + sample) * features) + feature];
}

auto launch_transpose(const float* source, float* destination, int batch, int time, int features, bool to_time_major)
    -> void
{
    const int count = batch * time * features;
    if (count <= 0)
    {
        return;
    }
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    if (to_time_major)
    {
        batch_to_time_kernel<<<blocks, threads, 0, dl::current_stream()>>>(source, destination, batch, time, features);
    }
    else
    {
        time_to_batch_kernel<<<blocks, threads, 0, dl::current_stream()>>>(source, destination, batch, time, features);
    }
    CHECK_CUDA(cudaGetLastError());
}

__global__ void scatter_paired_kernel(float* base, const float* values, int rows, int cols, int row_stride, int col_stride)
{
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int count = rows * cols;
    if (index >= count)
    {
        return;
    }
    const int row = index / cols;
    const int col = index - (row * cols);
    base[(row * row_stride) + (col * col_stride)] = values[index];
}

auto float_count_for_bytes(std::size_t bytes) -> int
{
    const std::size_t count = (bytes + sizeof(float) - 1U) / sizeof(float);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("cuDNN GRU buffer is too large");
    }
    return static_cast<int>(count);
}

auto require_contiguous_rank3(const dl::Tensor& tensor, const char* name) -> void
{
    if (tensor.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error(std::string(name) + " must reside on the GPU");
    }
    const std::vector<int>& shape = tensor.get_shape();
    const std::vector<int>& strides = tensor.get_strides();
    if (shape.size() != 3 || strides.size() != 3)
    {
        throw std::runtime_error(std::string(name) + " must have shape [N, T, F]");
    }
    if (strides[2] != 1 || strides[1] != shape[2] || strides[0] != shape[1] * shape[2])
    {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}

auto ensure_zero_like(std::optional<dl::Tensor>& slot, const dl::Tensor& like) -> dl::Tensor&
{
    if (!slot.has_value() || slot->get_shape() != like.get_shape())
    {
        slot = dl::Tensor(like.get_shape(), dl::Device::GPU, like.get_dtype());
        CHECK_CUDA(cudaMemsetAsync(slot->data(), 0, slot->nbytes(), dl::current_stream()));
    }
    return *slot;
}

auto copy_same_size(dl::Tensor& destination, const dl::Tensor& source, const char* name) -> void
{
    if (source.get_device() != dl::Device::GPU || destination.get_size() != source.get_size())
    {
        throw std::runtime_error(std::string(name) + " tensor size mismatch");
    }
    if (source.get_size() == 0)
    {
        return;
    }
    dl::memcpy_d2d_on_current(destination.data(), source.data(), source.nbytes());
}

}

struct Gru::Impl
{
    int input_size = 0;
    int hidden_size = 0;
    std::size_t weight_bytes = 0;
    std::size_t work_bytes = 0;
    std::size_t reserve_bytes = 0;
    int batch = 0;
    int time = 0;
    bool prepared = false;
    bool prepared_training = false;
    bool forward_ready = false;

    bool time_major = false;
    float padding_fill = 0.0F;
    std::vector<int> seq_lengths;

    struct Descriptor
    {
        cudnnRNNDescriptor_t rnn = nullptr;
        cudnnDropoutDescriptor_t dropout = nullptr;
        cudnnRNNDataDescriptor_t x = nullptr;
        cudnnRNNDataDescriptor_t y = nullptr;
        cudnnTensorDescriptor_t hidden = nullptr;

        Descriptor() = default;
        Descriptor(const Descriptor&) = delete;
        auto operator=(const Descriptor&) -> Descriptor& = delete;

        ~Descriptor()
        {
            if (hidden != nullptr)
            {
                cudnnDestroyTensorDescriptor(hidden);
            }
            if (y != nullptr)
            {
                cudnnDestroyRNNDataDescriptor(y);
            }
            if (x != nullptr)
            {
                cudnnDestroyRNNDataDescriptor(x);
            }
            if (rnn != nullptr)
            {
                cudnnDestroyRNNDescriptor(rnn);
            }
            if (dropout != nullptr)
            {
                cudnnDestroyDropoutDescriptor(dropout);
            }
        }
    };

    struct QueryTensor
    {
        cudnnTensorDescriptor_t handle = nullptr;

        QueryTensor()
        {
            cudnnTensorDescriptor_t created = nullptr;
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&created));
            handle = created;
        }

        QueryTensor(const QueryTensor&) = delete;
        auto operator=(const QueryTensor&) -> QueryTensor& = delete;

        ~QueryTensor()
        {
            if (handle != nullptr)
            {
                cudnnDestroyTensorDescriptor(handle);
            }
        }
    };

    dl::Tensor dropout_states;
    Descriptor desc;
    dl::Tensor weights;
    dl::Tensor weight_gradient;
    std::optional<dl::Tensor> weight_moment;
    std::optional<dl::Tensor> weight_second;
    std::optional<dl::Tensor> x;
    std::optional<dl::Tensor> y;
    std::optional<dl::Tensor> hy;
    std::optional<dl::Tensor> hx;
    std::optional<dl::Tensor> dx;
    std::optional<dl::Tensor> dx_batch;
    std::optional<dl::Tensor> dy;
    std::optional<dl::Tensor> workspace;
    std::optional<dl::Tensor> reserve;

    Impl(int input, int hidden)
        : input_size(input)
        , hidden_size(hidden)
    {
        if (input_size < 1 || hidden_size < 1)
        {
            throw std::runtime_error("Gru sizes must be positive");
        }

        const cudnnHandle_t handle = dl::get_cudnn_handle();
        std::size_t state_bytes = 0;
        CHECK_CUDNN(cudnnDropoutGetStatesSize(handle, &state_bytes));
        void* state_ptr = nullptr;
        if (state_bytes > 0)
        {
            dropout_states = dl::Tensor({ float_count_for_bytes(state_bytes) }, dl::Device::GPU);
            CHECK_CUDA(cudaMemset(dropout_states.data(), 0, dropout_states.nbytes()));
            state_ptr = dropout_states.data();
        }

        cudnnDropoutDescriptor_t dropout = nullptr;
        CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout));
        desc.dropout = dropout;
        CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout, handle, 0.0F, state_ptr, state_bytes, 42ULL));

        cudnnRNNDescriptor_t rnn = nullptr;
        CHECK_CUDNN(cudnnCreateRNNDescriptor(&rnn));
        desc.rnn = rnn;

        auto set_rnn = [&](cudnnRNNAlgo_t algo) -> cudnnStatus_t
        {
            return cudnnSetRNNDescriptor_v8(rnn, algo, CUDNN_GRU, CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, input_size, hidden_size,
                hidden_size, 1, dropout, CUDNN_RNN_PADDED_IO_ENABLED);
        };
        int device = 0;
        cudaDeviceProp properties {};
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
        if (hidden_size <= 128 && properties.major >= 6
            && set_rnn(CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H) == CUDNN_STATUS_SUCCESS)
        {
            time_major = true;
        }
        else
        {
            CHECK_CUDNN(set_rnn(CUDNN_RNN_ALGO_STANDARD));
            time_major = false;
        }

        cudnnRNNDataDescriptor_t x_desc = nullptr;
        cudnnRNNDataDescriptor_t y_desc = nullptr;
        cudnnTensorDescriptor_t h_desc = nullptr;
        CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&x_desc));
        CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&y_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&h_desc));
        desc.x = x_desc;
        desc.y = y_desc;
        desc.hidden = h_desc;

        allocate_weight_space(handle);
        if (time_major && !input_gates_match(handle))
        {

            time_major = false;
            CHECK_CUDNN(set_rnn(CUDNN_RNN_ALGO_STANDARD));
            allocate_weight_space(handle);
        }
        initialize_weights(handle);
    }

    auto allocate_weight_space(cudnnHandle_t handle) -> void
    {
        CHECK_CUDNN(cudnnGetRNNWeightSpaceSize(handle, desc.rnn, &weight_bytes));
        if (weight_bytes == 0)
        {
            throw std::runtime_error("cuDNN reported an empty GRU weight space");
        }
        weights = dl::Tensor({ float_count_for_bytes(weight_bytes) }, dl::Device::GPU);
        weight_gradient = dl::Tensor(weights.get_shape(), dl::Device::GPU);
        CHECK_CUDA(cudaMemset(weights.data(), 0, weights.nbytes()));
        CHECK_CUDA(cudaMemset(weight_gradient.data(), 0, weight_gradient.nbytes()));
    }

    auto input_gates_match(cudnnHandle_t handle) -> bool
    {
        auto matches = [&](int gate, int expect_cols) -> bool
        {
            QueryTensor matrix;
            QueryTensor bias;
            void* matrix_addr = nullptr;
            void* bias_addr = nullptr;
            if (cudnnGetRNNWeightParams(handle, desc.rnn, 0, weight_bytes, weights.data(), gate, matrix.handle,
                    &matrix_addr, bias.handle, &bias_addr)
                    != CUDNN_STATUS_SUCCESS
                || matrix_addr == nullptr)
            {
                return false;
            }
            cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
            int rank = 0;
            int dims[8] = {};
            int strides[8] = {};
            if (cudnnGetTensorNdDescriptor(matrix.handle, 8, &data_type, &rank, dims, strides) != CUDNN_STATUS_SUCCESS)
            {
                return false;
            }
            return rank >= 3 && dims[1] == hidden_size && dims[2] == expect_cols;
        };
        return matches(0, input_size) && matches(3, hidden_size);
    }

    auto initialize_weights(cudnnHandle_t handle) -> void
    {
        const float bound = 1.0F / std::sqrt(static_cast<float>(hidden_size));
        for (int gate = 0; gate < 6; ++gate)
        {
            QueryTensor matrix;
            QueryTensor bias;
            void* matrix_addr = nullptr;
            void* bias_addr = nullptr;
            CHECK_CUDNN(cudnnGetRNNWeightParams(handle, desc.rnn, 0, weight_bytes, weights.data(), gate, matrix.handle,
                &matrix_addr, bias.handle, &bias_addr));
            fill_parameter(matrix.handle, matrix_addr, bound, 0xC0FFEEULL + (static_cast<unsigned long long>(gate) * 2ULL));
            fill_parameter(bias.handle, bias_addr, bound,
                0xBEEFULL + (static_cast<unsigned long long>(gate) * 2ULL) + 1ULL);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    static auto fill_parameter(cudnnTensorDescriptor_t descriptor, void* address, float bound, unsigned long long seed)
        -> void
    {
        if (address == nullptr)
        {
            return;
        }
        cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
        int rank = 0;
        int dims[8] = {};
        int strides[8] = {};
        CHECK_CUDNN(cudnnGetTensorNdDescriptor(descriptor, 8, &data_type, &rank, dims, strides));
        if (rank < 3 || dims[1] <= 0 || dims[2] <= 0)
        {
            return;
        }
        const int rows = dims[1];
        const int cols = dims[2];
        const int count = rows * cols;
        std::vector<float> host(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index)
        {
            host[static_cast<std::size_t>(index)] = dl::paired_uniform(index, -bound, bound, seed);
        }
        dl::Tensor staging({ count }, dl::Device::GPU);
        CHECK_CUDA(cudaMemcpy(staging.data(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
        const dim3 grid(static_cast<unsigned int>((count + kFillThreads - 1) / kFillThreads));
        scatter_paired_kernel<<<grid, kFillThreads>>>(static_cast<float*>(address), staging.data(), rows, cols,
            strides[1], strides[2]);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    auto prepare(int next_batch, int next_time, bool training) -> void
    {
        if (next_batch < 1 || next_time < 1 || next_time > 65535)
        {
            throw std::runtime_error("Gru sequence dimensions are out of range");
        }

        if (prepared && batch == next_batch && time == next_time && prepared_training == training)
        {
            return;
        }
        seq_lengths.assign(static_cast<std::size_t>(next_batch), next_time);
        const auto layout = time_major ? CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
                                       : CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
        CHECK_CUDNN(cudnnSetRNNDataDescriptor(desc.x, CUDNN_DATA_FLOAT, layout, next_time, next_batch, input_size,
            seq_lengths.data(), &padding_fill));
        CHECK_CUDNN(cudnnSetRNNDataDescriptor(desc.y, CUDNN_DATA_FLOAT, layout, next_time, next_batch, hidden_size,
            seq_lengths.data(), &padding_fill));

        const int dims[3] = { 1, next_batch, hidden_size };
        const int strides[3] = { next_batch * hidden_size, hidden_size, 1 };
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc.hidden, CUDNN_DATA_FLOAT, 3, dims, strides));

        const cudnnForwardMode_t mode = training ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;
        CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(dl::get_cudnn_handle(), desc.rnn, mode, desc.x, &work_bytes, &reserve_bytes));
        ensure_bytes(workspace, work_bytes);
        ensure_bytes(reserve, reserve_bytes);
        if (time_major)
        {
            dl::Tensor::ensure(x, { next_time, next_batch, input_size }, dl::Device::GPU);
            dl::Tensor::ensure(y, { next_time, next_batch, hidden_size }, dl::Device::GPU);
            dl::Tensor::ensure(dx, { next_time, next_batch, input_size }, dl::Device::GPU);
            dl::Tensor::ensure(dx_batch, { next_batch, next_time, input_size }, dl::Device::GPU);
            dl::Tensor::ensure(dy, { next_time, next_batch, hidden_size }, dl::Device::GPU);
        }
        else
        {
            dl::Tensor::ensure(x, { next_batch, next_time, input_size }, dl::Device::GPU);
            dl::Tensor::ensure(y, { next_batch, next_time, hidden_size }, dl::Device::GPU);
            dl::Tensor::ensure(dx, { next_batch, next_time, input_size }, dl::Device::GPU);
            dl::Tensor::ensure(dy, { next_batch, next_time, hidden_size }, dl::Device::GPU);
        }
        dl::Tensor::ensure(hy, { 1, next_batch, hidden_size }, dl::Device::GPU);
        dl::Tensor::ensure(hx, { 1, next_batch, hidden_size }, dl::Device::GPU);

        CHECK_CUDA(cudaMemsetAsync(hx->data(), 0, hx->nbytes(), dl::current_stream()));
        batch = next_batch;
        time = next_time;
        prepared_training = training;
        prepared = true;
    }

    static auto ensure_bytes(std::optional<dl::Tensor>& slot, std::size_t bytes) -> void
    {
        if (bytes == 0)
        {
            slot.reset();
            return;
        }
        dl::Tensor::ensure(slot, { float_count_for_bytes(bytes) }, dl::Device::GPU);
    }

    [[nodiscard]] auto buffer_ptr(std::optional<dl::Tensor>& slot, std::size_t bytes) -> void*
    {
        if (bytes == 0 || !slot.has_value())
        {
            return nullptr;
        }
        return slot->data();
    }
};

Gru::Gru(int input_size, int hidden_size)
    : impl_(std::make_unique<Impl>(input_size, hidden_size))
{
    device_ = dl::Device::GPU;
}

Gru::~Gru() = default;

auto Gru::input_size() const -> int
{
    return impl_->input_size;
}

auto Gru::hidden_size() const -> int
{
    return impl_->hidden_size;
}

auto Gru::forward(const dl::Tensor& input_tensor, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Gru_Forward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    require_contiguous_rank3(input_tensor, "Gru::forward input");
    const int batch = input_tensor.get_shape()[0];
    const int time = input_tensor.get_shape()[1];
    if (input_tensor.get_shape()[2] != impl_->input_size)
    {
        throw std::runtime_error("Gru::forward feature count does not match the layer");
    }

    impl_->prepare(batch, time, is_training_);
    if (impl_->time_major)
    {
        launch_transpose(input_tensor.data(), impl_->x->data(), batch, time, impl_->input_size, true);
    }
    else
    {
        dl::memcpy_d2d_on_current(impl_->x->data(), input_tensor.data(), input_tensor.nbytes());
    }
    const cudnnForwardMode_t mode = is_training_ ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;
    CHECK_CUDNN(cudnnRNNForward(dl::get_cudnn_handle(), impl_->desc.rnn, mode, nullptr, impl_->desc.x, impl_->x->data(),
        impl_->desc.y, impl_->y->data(), impl_->desc.hidden, impl_->hx->data(), impl_->hy->data(), nullptr, nullptr,
        nullptr, impl_->weight_bytes, impl_->weights.data(), impl_->work_bytes,
        impl_->buffer_ptr(impl_->workspace, impl_->work_bytes), impl_->reserve_bytes,
        impl_->buffer_ptr(impl_->reserve, impl_->reserve_bytes)));
    impl_->forward_ready = is_training_;
    return impl_->hy->view({ batch, impl_->hidden_size });
}

auto Gru::backward(const dl::Tensor& output_error_derivative, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("Gru_Backward");
    const dl::StreamGuard stream_guard(stream);
    dl::bind_cudnn_stream(stream);
    if (!impl_->forward_ready || !impl_->prepared)
    {
        throw std::runtime_error("Gru::backward requires a training forward");
    }
    if (output_error_derivative.get_device() != dl::Device::GPU || output_error_derivative.get_shape().size() != 2
        || output_error_derivative.get_shape()[0] != impl_->batch
        || output_error_derivative.get_shape()[1] != impl_->hidden_size)
    {
        throw std::runtime_error("Gru::backward expects the final-state gradient [N, H]");
    }

    CHECK_CUDA(cudaMemsetAsync(impl_->dy->data(), 0, impl_->dy->nbytes(), dl::current_stream()));
    CHECK_CUDA(cudaMemsetAsync(impl_->weight_gradient.data(), 0, impl_->weight_bytes, dl::current_stream()));
    CHECK_CUDNN(cudnnRNNBackwardData_v8(dl::get_cudnn_handle(), impl_->desc.rnn, nullptr, impl_->desc.y, impl_->y->data(),
        impl_->dy->data(), impl_->desc.x, impl_->dx->data(), impl_->desc.hidden, impl_->hx->data(),
        output_error_derivative.data(), nullptr, nullptr, nullptr, nullptr, nullptr, impl_->weight_bytes,
        impl_->weights.data(), impl_->work_bytes, impl_->buffer_ptr(impl_->workspace, impl_->work_bytes),
        impl_->reserve_bytes, impl_->buffer_ptr(impl_->reserve, impl_->reserve_bytes)));
    CHECK_CUDNN(cudnnRNNBackwardWeights_v8(dl::get_cudnn_handle(), impl_->desc.rnn, CUDNN_WGRAD_MODE_ADD, nullptr,
        impl_->desc.x, impl_->x->data(), impl_->desc.hidden, impl_->hx->data(), impl_->desc.y, impl_->y->data(),
        impl_->weight_bytes, impl_->weight_gradient.data(), impl_->work_bytes,
        impl_->buffer_ptr(impl_->workspace, impl_->work_bytes), impl_->reserve_bytes,
        impl_->buffer_ptr(impl_->reserve, impl_->reserve_bytes)));
    impl_->forward_ready = false;
    if (!impl_->time_major)
    {
        return impl_->dx->as_view();
    }
    launch_transpose(impl_->dx->data(), impl_->dx_batch->data(), impl_->batch, impl_->time, impl_->input_size, false);
    return impl_->dx_batch->as_view();
}

void Gru::step(cudaStream_t stream)
{
    const dl::NvtxRange nvtx_range("Gru_Step");
    const dl::StreamGuard stream_guard(stream);
    const float clip = parameter_clip_bound();
    const float lr = step_learning_rate();
    if (use_adam)
    {
        ++adam_step;
        dl::Tensor& moment = ensure_zero_like(impl_->weight_moment, impl_->weights);
        dl::Tensor& second = ensure_zero_like(impl_->weight_second, impl_->weights);
        impl_->weights.adam_update_(impl_->weight_gradient, moment, second, lr, 0.9F, 0.999F, weight_decay, 1.0e-8F,
            adam_step, clip);
        return;
    }
    if (momentum > 0.0F)
    {
        dl::Tensor& velocity = ensure_zero_like(impl_->weight_moment, impl_->weights);
        impl_->weights.sgd_momentum_update_(impl_->weight_gradient, velocity, lr, momentum, weight_decay, clip);
        return;
    }
    impl_->weights.sgd_update_(impl_->weight_gradient, lr, weight_decay, clip);
}

void Gru::clip_gradients(float abs_bound, cudaStream_t stream)
{
    const dl::StreamGuard stream_guard(stream);
    if (abs_bound <= 0.0F)
    {
        return;
    }
    impl_->weight_gradient.clamp_(-abs_bound, abs_bound);
}

auto Gru::get_parameters() -> std::map<std::string, dl::Tensor>
{
    std::map<std::string, dl::Tensor> params;
    params.emplace("weight_space", impl_->weights.view(impl_->weights.get_shape()));
    return params;
}

void Gru::set_parameters(const std::map<std::string, dl::Tensor>& params)
{
    copy_same_size(impl_->weights, params.at("weight_space"), "Gru::set_parameters weight_space");
}

auto Gru::to(dl::Device device) -> void
{
    if (device != dl::Device::GPU)
    {
        throw std::runtime_error("Gru parameters must remain on the GPU");
    }
    device_ = device;
}

auto Gru::export_linear_layers() const -> std::vector<LinearLayer>
{
    CHECK_CUDA(cudaDeviceSynchronize());
    const std::size_t words = impl_->weights.get_size();
    std::vector<float> space(words);
    if (words > 0)
    {
        CHECK_CUDA(cudaMemcpy(space.data(), impl_->weights.data(), impl_->weights.nbytes(), cudaMemcpyDeviceToHost));
    }
    const auto* device_base = static_cast<const float*>(impl_->weights.data());
    const cudnnHandle_t handle = dl::get_cudnn_handle();

    auto read_tensor = [&](cudnnTensorDescriptor_t descriptor, void* address, int& rows, int& cols) -> std::vector<float>
    {
        rows = 0;
        cols = 0;
        if (address == nullptr || descriptor == nullptr)
        {
            return {};
        }
        cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
        int rank = 0;
        int dims[8] = {};
        int strides[8] = {};
        CHECK_CUDNN(cudnnGetTensorNdDescriptor(descriptor, 8, &data_type, &rank, dims, strides));
        int row_stride = 1;
        int col_stride = 1;
        if (rank >= 3 && dims[1] > 0 && dims[2] > 0)
        {
            rows = dims[1];
            cols = dims[2];
            row_stride = strides[1];
            col_stride = strides[2];
        }
        else if (rank == 2 && dims[0] > 0 && dims[1] > 0)
        {
            rows = dims[0];
            cols = dims[1];
            row_stride = strides[0];
            col_stride = strides[1];
        }
        else if (rank == 1 && dims[0] > 0)
        {
            rows = dims[0];
            cols = 1;
            row_stride = strides[0] == 0 ? 1 : strides[0];
            col_stride = 0;
        }
        else
        {
            return {};
        }

        const auto offset = static_cast<const float*>(address) - device_base;
        if (offset < 0 || static_cast<std::size_t>(offset) >= space.size())
        {
            throw std::runtime_error("GRU linear layer lies outside the weight space");
        }
        std::vector<float> values(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                const std::ptrdiff_t index = static_cast<std::ptrdiff_t>(offset)
                    + (static_cast<std::ptrdiff_t>(row) * row_stride) + (static_cast<std::ptrdiff_t>(col) * col_stride);
                if (index < 0 || static_cast<std::size_t>(index) >= space.size())
                {
                    throw std::runtime_error("GRU linear layer stride walks off the weight space");
                }
                values[(static_cast<std::size_t>(row) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(col)]
                    = space[static_cast<std::size_t>(index)];
            }
        }
        return values;
    };

    std::vector<LinearLayer> layers;
    layers.reserve(6);
    for (int gate = 0; gate < 6; ++gate)
    {
        Impl::QueryTensor matrix;
        Impl::QueryTensor bias;
        void* matrix_addr = nullptr;
        void* bias_addr = nullptr;
        CHECK_CUDNN(cudnnGetRNNWeightParams(handle, impl_->desc.rnn, 0, impl_->weight_bytes,
            const_cast<float*>(impl_->weights.data()), gate, matrix.handle, &matrix_addr, bias.handle, &bias_addr));
        LinearLayer layer;
        int bias_rows = 0;
        int bias_cols = 0;
        layer.matrix = read_tensor(matrix.handle, matrix_addr, layer.rows, layer.cols);
        layer.bias = read_tensor(bias.handle, bias_addr, bias_rows, bias_cols);
        layers.push_back(std::move(layer));
    }
    return layers;
}
