#include "DeepLearnLib/DarknetDetectionLoss.hpp"
#include "DeepLearnLib/Nvtx.hpp"
#include "DeepLearnLib/Precision.hpp"
#include "DeepLearnLib/SafeMath.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int kThreads = 256;

__device__ auto clampf(float value, float lo) -> float
{
    return value < lo ? lo : value;
}

__device__ auto box_iou(float cx1, float cy1, float w1, float h1, float cx2, float cy2, float w2, float h2) -> float
{
    const float b1_x1 = cx1 - (w1 * 0.5F);
    const float b1_y1 = cy1 - (h1 * 0.5F);
    const float b1_x2 = cx1 + (w1 * 0.5F);
    const float b1_y2 = cy1 + (h1 * 0.5F);
    const float b2_x1 = cx2 - (w2 * 0.5F);
    const float b2_y1 = cy2 - (h2 * 0.5F);
    const float b2_x2 = cx2 + (w2 * 0.5F);
    const float b2_y2 = cy2 + (h2 * 0.5F);
    const float inter_w = clampf(fminf(b1_x2, b2_x2) - fmaxf(b1_x1, b2_x1), 0.0F);
    const float inter_h = clampf(fminf(b1_y2, b2_y2) - fmaxf(b1_y1, b2_y1), 0.0F);
    const float area1 = clampf(w1 * h1, dl::kSafeEps);
    const float area2 = clampf(w2 * h2, dl::kSafeEps);
    return dl::safe_div(inter_w * inter_h, area1 + area2 - (inter_w * inter_h));
}

struct DetectionWorkspace
{
    std::optional<dl::Tensor> cell_loss;
    std::optional<dl::Tensor> scalar;
    std::optional<dl::Tensor> grad;
};

auto workspace() -> DetectionWorkspace&
{
    static DetectionWorkspace buffers;
    return buffers;
}

auto pred_length(const DarknetDetectionLoss::Config& config) -> int
{
    const int cells = config.side * config.side;
    return cells * (config.num_classes + config.num_boxes + (config.num_boxes * config.coords));
}

auto truth_length(const DarknetDetectionLoss::Config& config) -> int
{
    return config.side * config.side * (1 + config.coords + config.num_classes);
}

auto require_gpu_pair(const dl::Tensor& target, const dl::Tensor& prediction) -> void
{
    if (target.get_device() != dl::Device::GPU || prediction.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error("DarknetDetectionLoss requires GPU tensors");
    }
}

__global__ void darknet_detection_kernel(const float* pred, const float* tgt, float* grad, float* cell_loss, int batch,
    int side, int num_boxes, int coords, int classes, int write_grad, int sqrt_wh, int rescore, float object_scale,
    float noobject_scale, float class_scale, float coord_scale, float inv_batch)
{
    const int cells = side * side;
    const int idx = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (idx >= batch * cells)
    {
        return;
    }

    const int b = idx / cells;
    const int cell = idx % cells;
    const int row = cell / side;
    const int col = cell % side;
    const int pred_stride = cells * (classes + num_boxes + (num_boxes * coords));
    const int truth_stride = cells * (1 + coords + classes);
    const float* p = pred + (b * pred_stride);
    const float* t = tgt + (b * truth_stride) + (cell * (1 + coords + classes));
    float* g = grad + (b * pred_stride);

    float loss = 0.0F;
    const float is_obj = t[0];

    for (int box_idx = 0; box_idx < num_boxes; ++box_idx)
    {
        const int p_index = (cells * classes) + (cell * num_boxes) + box_idx;
        const float objectness = p[p_index];
        loss += noobject_scale * objectness * objectness;
        if (write_grad != 0)
        {
            g[p_index] = noobject_scale * objectness * inv_batch;
        }
    }

    if (is_obj <= 0.0F)
    {
        cell_loss[idx] = loss;
        return;
    }

    const int class_index = cell * classes;
    for (int class_idx = 0; class_idx < classes; ++class_idx)
    {
        const float diff = p[class_index + class_idx] - t[1 + class_idx];
        loss += class_scale * diff * diff;
        if (write_grad != 0)
        {
            g[class_index + class_idx] = class_scale * diff * inv_batch;
        }
    }

    const float truth_x = t[1 + classes] / static_cast<float>(side);
    const float truth_y = t[1 + classes + 1] / static_cast<float>(side);
    const float truth_w = t[1 + classes + 2];
    const float truth_h = t[1 + classes + 3];

    int best_index = 0;
    float best_iou = 0.0F;
    float best_rmse = 1.0e6F;
    for (int box_idx = 0; box_idx < num_boxes; ++box_idx)
    {
        const int box_index = (cells * (classes + num_boxes)) + (((cell * num_boxes) + box_idx) * coords);
        float pred_w = p[box_index + 2];
        float pred_h = p[box_index + 3];
        if (sqrt_wh != 0)
        {
            pred_w = pred_w * pred_w;
            pred_h = pred_h * pred_h;
        }
        const float pred_x = (p[box_index + 0] + static_cast<float>(col)) / static_cast<float>(side);
        const float pred_y = (p[box_index + 1] + static_cast<float>(row)) / static_cast<float>(side);
        const float iou = box_iou(pred_x, pred_y, pred_w, pred_h, truth_x, truth_y, truth_w, truth_h);
        const float dx = pred_x - truth_x;
        const float dy = pred_y - truth_y;
        const float dw = pred_w - truth_w;
        const float dh = pred_h - truth_h;
        const float rmse = sqrtf((dx * dx) + (dy * dy) + (dw * dw) + (dh * dh));
        if (best_iou > 0.0F || iou > 0.0F)
        {
            if (iou > best_iou)
            {
                best_iou = iou;
                best_index = box_idx;
            }
        }
        else if (rmse < best_rmse)
        {
            best_rmse = rmse;
            best_index = box_idx;
        }
    }

    const int best_p = (cells * classes) + (cell * num_boxes) + best_index;
    const int best_box = (cells * (classes + num_boxes)) + (((cell * num_boxes) + best_index) * coords);
    float pred_w = p[best_box + 2];
    float pred_h = p[best_box + 3];
    if (sqrt_wh != 0)
    {
        pred_w = pred_w * pred_w;
        pred_h = pred_h * pred_h;
    }
    const float pred_x = (p[best_box + 0] + static_cast<float>(col)) / static_cast<float>(side);
    const float pred_y = (p[best_box + 1] + static_cast<float>(row)) / static_cast<float>(side);
    const float iou = box_iou(pred_x, pred_y, pred_w, pred_h, truth_x, truth_y, truth_w, truth_h);
    const float object_target = rescore != 0 ? iou : 1.0F;
    const float objectness = p[best_p];
    loss -= noobject_scale * objectness * objectness;
    loss += object_scale * (object_target - objectness) * (object_target - objectness);
    loss += (1.0F - iou) * (1.0F - iou);
    if (write_grad != 0)
    {
        g[best_p] = object_scale * (objectness - object_target) * inv_batch;
        g[best_box + 0] = coord_scale * (p[best_box + 0] - t[1 + classes + 0]) * inv_batch;
        g[best_box + 1] = coord_scale * (p[best_box + 1] - t[1 + classes + 1]) * inv_batch;
        if (sqrt_wh != 0)
        {
            g[best_box + 2] = coord_scale * (p[best_box + 2] - sqrtf(clampf(truth_w, 0.0F))) * inv_batch;
            g[best_box + 3] = coord_scale * (p[best_box + 3] - sqrtf(clampf(truth_h, 0.0F))) * inv_batch;
        }
        else
        {
            g[best_box + 2] = coord_scale * (p[best_box + 2] - truth_w) * inv_batch;
            g[best_box + 3] = coord_scale * (p[best_box + 3] - truth_h) * inv_batch;
        }
    }
    cell_loss[idx] = loss;
}

__global__ void mean_loss_kernel(const float* cell_loss, float* mean_loss, int cell_count, float inv_batch)
{
    __shared__ float shared_sum[kThreads];
    float partial = 0.0F;
    for (int index = static_cast<int>(threadIdx.x); index < cell_count; index += static_cast<int>(blockDim.x))
    {
        partial += cell_loss[index];
    }
    shared_sum[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1)
    {
        if (static_cast<int>(threadIdx.x) < stride)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        mean_loss[0] = shared_sum[0] * inv_batch;
    }
}

auto flatten_pred(const dl::Tensor& prediction, int expected) -> dl::Tensor
{
    if (prediction.get_size() % static_cast<size_t>(expected) != 0)
    {
        throw std::runtime_error("DarknetDetectionLoss prediction size mismatch, expected multiple of "
            + std::to_string(expected) + " got " + prediction.describe());
    }
    const int batch = static_cast<int>(prediction.get_size() / static_cast<size_t>(expected));
    return prediction.view({ batch, expected });
}

auto flatten_truth(const dl::Tensor& target, int expected) -> dl::Tensor
{
    if (target.get_size() % static_cast<size_t>(expected) != 0)
    {
        throw std::runtime_error("DarknetDetectionLoss target size mismatch, expected multiple of "
            + std::to_string(expected) + " got " + target.describe());
    }
    const int batch = static_cast<int>(target.get_size() / static_cast<size_t>(expected));
    return target.view({ batch, expected });
}

auto launch_cells(int count) -> dim3
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

auto run_detection(const dl::Tensor& target, const dl::Tensor& prediction, const DarknetDetectionLoss::Config& config,
    bool write_grad, cudaStream_t stream) -> dl::Tensor
{
    require_gpu_pair(target, prediction);
    if (config.side <= 0 || config.num_boxes <= 0 || config.coords != 4 || config.num_classes <= 0)
    {
        throw std::runtime_error("DarknetDetectionLoss received invalid config");
    }

    const int pred_len = pred_length(config);
    const int truth_len = truth_length(config);
    const dl::Tensor pred_f32 = prediction.to_dtype(dl::Dtype::Float32, stream);
    const dl::Tensor tgt_f32 = target.to_dtype(dl::Dtype::Float32, stream);
    const dl::Tensor pred = flatten_pred(pred_f32, pred_len);
    const dl::Tensor tgt = flatten_truth(tgt_f32, truth_len);
    if (pred.get_shape()[0] != tgt.get_shape()[0])
    {
        throw std::runtime_error("DarknetDetectionLoss batch sizes do not match");
    }

    const int batch = pred.get_shape()[0];
    const int cell_count = batch * config.side * config.side;
    const float inv_batch = dl::safe_inv(static_cast<float>(batch));
    dl::Tensor& cell_loss = dl::Tensor::ensure(workspace().cell_loss, { cell_count }, dl::Device::GPU);
    dl::Tensor& grad = dl::Tensor::ensure(workspace().grad, pred.get_shape(), dl::Device::GPU);
    if (write_grad)
    {
        CHECK_CUDA(cudaMemsetAsync(grad.data(), 0, grad.nbytes(), stream));
    }

    darknet_detection_kernel<<<launch_cells(cell_count), kThreads, 0, stream>>>(pred.data(), tgt.data(), grad.data(),
        cell_loss.data(), batch, config.side, config.num_boxes, config.coords, config.num_classes,
        write_grad ? 1 : 0, config.sqrt_wh ? 1 : 0, config.rescore ? 1 : 0, config.object_scale,
        config.noobject_scale, config.class_scale, config.coord_scale, inv_batch);
    CHECK_CUDA(cudaGetLastError());

    if (write_grad)
    {
        const float scale = dl::loss_scale();
        if (scale != 1.0F)
        {
            grad.mul_(scale);
        }
        return grad.as_view();
    }

    dl::Tensor& mean_loss = dl::Tensor::ensure(workspace().scalar, { 1 }, dl::Device::GPU);
    mean_loss_kernel<<<1, kThreads, 0, stream>>>(cell_loss.data(), mean_loss.data(), cell_count, inv_batch);
    CHECK_CUDA(cudaGetLastError());
    return mean_loss.as_view();
}

} // namespace

auto DarknetDetectionLoss::loss(const dl::Tensor& target, const dl::Tensor& prediction, const Config& config,
    cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("DarknetDetectionLoss_Loss");
    const dl::StreamGuard stream_guard(stream);
    return run_detection(target, prediction, config, false, stream);
}

auto DarknetDetectionLoss::loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction,
    const Config& config, cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("DarknetDetectionLoss_LossDerivative");
    const dl::StreamGuard stream_guard(stream);
    return run_detection(target, prediction, config, true, stream);
}
