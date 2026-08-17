#include "DeepLearnLib/YOLOLoss.hpp"
#include "DeepLearnLib/Nvtx.hpp"
#include "DeepLearnLib/Precision.hpp"
#include "DeepLearnLib/SafeMath.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int kGridSize = 7;
constexpr int kCellsPerImage = kGridSize * kGridSize;
constexpr int kBoxAttrs = 4;
constexpr int kThreads = 256;
constexpr float kLambdaCoord = 5.0F;
constexpr float kLambdaNoobj = 0.5F;

__host__ __device__ auto ceil_div(int value, int divisor) -> int
{
    return (value + divisor - 1) / divisor;
}

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
    const float inter_area = inter_w * inter_h;

    const float area1 = clampf(w1 * h1, dl::kSafeEps);
    const float area2 = clampf(w2 * h2, dl::kSafeEps);
    return dl::safe_div(inter_area, area1 + area2 - inter_area);
}

__device__ auto decode_center(float raw, int grid_index) -> float
{
    return (raw + static_cast<float>(grid_index)) / static_cast<float>(kGridSize);
}

__device__ auto cell_base(int batch, int row, int col, int final_dim) -> int
{
    return ((((batch * kGridSize) + row) * kGridSize) + col) * final_dim;
}

__global__ void yolo_iou_kernel(const float* box1, const float* box2, float* iou, int box_count)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= box_count)
    {
        return;
    }

    const int offset = idx * kBoxAttrs;
    iou[idx] = box_iou(box1[offset + 0], box1[offset + 1], box1[offset + 2], box1[offset + 3], box2[offset + 0],
        box2[offset + 1], box2[offset + 2], box2[offset + 3]);
}

__global__ void yolo_loss_forward_kernel(const float* pred, const float* tgt, float* cell_loss, int batch_size,
    int final_dim)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int cell_count = batch_size * kCellsPerImage;
    if (idx >= cell_count)
    {
        return;
    }

    const int batch = idx / kCellsPerImage;
    const int cell = idx % kCellsPerImage;
    const int row = cell / kGridSize;
    const int col = cell % kGridSize;
    const int base = cell_base(batch, row, col, final_dim);

    const float p1_x = pred[base + 0];
    const float p1_y = pred[base + 1];
    const float p1_w = pred[base + 2];
    const float p1_h = pred[base + 3];
    const float p1_c = pred[base + 4];
    const float p2_x = pred[base + 5];
    const float p2_y = pred[base + 6];
    const float p2_w = pred[base + 7];
    const float p2_h = pred[base + 8];
    const float p2_c = pred[base + 9];

    const float t_x = tgt[base + 0];
    const float t_y = tgt[base + 1];
    const float t_w = tgt[base + 2];
    const float t_h = tgt[base + 3];
    const float obj_mask = tgt[base + 4];

    const float iou1 = box_iou(decode_center(p1_x, col), decode_center(p1_y, row), p1_w, p1_h,
        decode_center(t_x, col), decode_center(t_y, row), t_w, t_h);
    const float iou2 = box_iou(decode_center(p2_x, col), decode_center(p2_y, row), p2_w, p2_h,
        decode_center(t_x, col), decode_center(t_y, row), t_w, t_h);

    const float box2_better = iou2 > iou1 ? 1.0F : 0.0F;
    const float resp_b1 = (1.0F - box2_better) * obj_mask;
    const float resp_b2 = box2_better * obj_mask;
    const float noobj_b1 = 1.0F - resp_b1;
    const float noobj_b2 = 1.0F - resp_b2;

    const float sqrt_p1_w = dl::safe_sqrt(p1_w);
    const float sqrt_p1_h = dl::safe_sqrt(p1_h);
    const float sqrt_p2_w = dl::safe_sqrt(p2_w);
    const float sqrt_p2_h = dl::safe_sqrt(p2_h);
    const float sqrt_t_w = dl::safe_sqrt(t_w);
    const float sqrt_t_h = dl::safe_sqrt(t_h);

    const float xy_b1 = ((p1_x - t_x) * (p1_x - t_x)) + ((p1_y - t_y) * (p1_y - t_y));
    const float xy_b2 = ((p2_x - t_x) * (p2_x - t_x)) + ((p2_y - t_y) * (p2_y - t_y));
    const float wh_b1 = ((sqrt_p1_w - sqrt_t_w) * (sqrt_p1_w - sqrt_t_w)) + ((sqrt_p1_h - sqrt_t_h) * (sqrt_p1_h - sqrt_t_h));
    const float wh_b2 = ((sqrt_p2_w - sqrt_t_w) * (sqrt_p2_w - sqrt_t_w)) + ((sqrt_p2_h - sqrt_t_h) * (sqrt_p2_h - sqrt_t_h));
    const float l_coord = kLambdaCoord * ((xy_b1 * resp_b1) + (xy_b2 * resp_b2) + (wh_b1 * resp_b1) + (wh_b2 * resp_b2));

    const float conf_obj = ((p1_c - iou1) * (p1_c - iou1) * resp_b1) + ((p2_c - iou2) * (p2_c - iou2) * resp_b2);
    const float conf_noobj = kLambdaNoobj * ((p1_c * p1_c * noobj_b1) + (p2_c * p2_c * noobj_b2));
    const float l_conf = conf_obj + conf_noobj;

    float l_class = 0.0F;
    const int num_classes = final_dim - 10;
    for (int class_idx = 0; class_idx < num_classes; ++class_idx)
    {
        const float diff = pred[base + 10 + class_idx] - tgt[base + 10 + class_idx];
        l_class += diff * diff;
    }
    l_class *= obj_mask;

    cell_loss[idx] = l_coord + l_conf + l_class;
}

__global__ void yolo_loss_backward_kernel(const float* pred, const float* tgt, float* grad, int batch_size,
    int final_dim, float inv_batch)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int cell_count = batch_size * kCellsPerImage;
    if (idx >= cell_count)
    {
        return;
    }

    const int batch = idx / kCellsPerImage;
    const int cell = idx % kCellsPerImage;
    const int row = cell / kGridSize;
    const int col = cell % kGridSize;
    const int base = cell_base(batch, row, col, final_dim);

    const float p1_x = pred[base + 0];
    const float p1_y = pred[base + 1];
    const float p1_w = pred[base + 2];
    const float p1_h = pred[base + 3];
    const float p1_c = pred[base + 4];
    const float p2_x = pred[base + 5];
    const float p2_y = pred[base + 6];
    const float p2_w = pred[base + 7];
    const float p2_h = pred[base + 8];
    const float p2_c = pred[base + 9];

    const float t_x = tgt[base + 0];
    const float t_y = tgt[base + 1];
    const float t_w = tgt[base + 2];
    const float t_h = tgt[base + 3];
    const float obj_mask = tgt[base + 4];

    const float iou1 = box_iou(decode_center(p1_x, col), decode_center(p1_y, row), p1_w, p1_h,
        decode_center(t_x, col), decode_center(t_y, row), t_w, t_h);
    const float iou2 = box_iou(decode_center(p2_x, col), decode_center(p2_y, row), p2_w, p2_h,
        decode_center(t_x, col), decode_center(t_y, row), t_w, t_h);

    const float box2_better = iou2 > iou1 ? 1.0F : 0.0F;
    const float resp_b1 = (1.0F - box2_better) * obj_mask;
    const float resp_b2 = box2_better * obj_mask;
    const float noobj_b1 = 1.0F - resp_b1;
    const float noobj_b2 = 1.0F - resp_b2;

    grad[base + 0] = 2.0F * kLambdaCoord * (p1_x - t_x) * resp_b1 * inv_batch;
    grad[base + 1] = 2.0F * kLambdaCoord * (p1_y - t_y) * resp_b1 * inv_batch;
    grad[base + 5] = 2.0F * kLambdaCoord * (p2_x - t_x) * resp_b2 * inv_batch;
    grad[base + 6] = 2.0F * kLambdaCoord * (p2_y - t_y) * resp_b2 * inv_batch;

    const float sqrt_p1_w = dl::safe_sqrt(p1_w);
    const float sqrt_p1_h = dl::safe_sqrt(p1_h);
    const float sqrt_p2_w = dl::safe_sqrt(p2_w);
    const float sqrt_p2_h = dl::safe_sqrt(p2_h);
    const float sqrt_t_w = dl::safe_sqrt(t_w);
    const float sqrt_t_h = dl::safe_sqrt(t_h);
    const float mask_p1_w = p1_w > dl::kSafeEps ? 1.0F : 0.0F;
    const float mask_p1_h = p1_h > dl::kSafeEps ? 1.0F : 0.0F;
    const float mask_p2_w = p2_w > dl::kSafeEps ? 1.0F : 0.0F;
    const float mask_p2_h = p2_h > dl::kSafeEps ? 1.0F : 0.0F;

    grad[base + 2] = kLambdaCoord * dl::safe_div(sqrt_p1_w - sqrt_t_w, sqrt_p1_w) * mask_p1_w * resp_b1 * inv_batch;
    grad[base + 3] = kLambdaCoord * dl::safe_div(sqrt_p1_h - sqrt_t_h, sqrt_p1_h) * mask_p1_h * resp_b1 * inv_batch;
    grad[base + 7] = kLambdaCoord * dl::safe_div(sqrt_p2_w - sqrt_t_w, sqrt_p2_w) * mask_p2_w * resp_b2 * inv_batch;
    grad[base + 8] = kLambdaCoord * dl::safe_div(sqrt_p2_h - sqrt_t_h, sqrt_p2_h) * mask_p2_h * resp_b2 * inv_batch;

    grad[base + 4] = ((2.0F * (p1_c - iou1) * resp_b1) + (2.0F * kLambdaNoobj * p1_c * noobj_b1)) * inv_batch;
    grad[base + 9] = ((2.0F * (p2_c - iou2) * resp_b2) + (2.0F * kLambdaNoobj * p2_c * noobj_b2)) * inv_batch;

    const int num_classes = final_dim - 10;
    for (int class_idx = 0; class_idx < num_classes; ++class_idx)
    {
        const float diff = pred[base + 10 + class_idx] - tgt[base + 10 + class_idx];
        grad[base + 10 + class_idx] = 2.0F * diff * obj_mask * inv_batch;
    }
}

__global__ void yolo_mean_loss_kernel(const float* cell_loss, float* mean_loss, int cell_count, float inv_batch)
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

auto launch_config(int count) -> dim3
{
    return dim3(static_cast<unsigned int>(std::max(1, ceil_div(count, kThreads))));
}

auto require_gpu_pair(const dl::Tensor& target, const dl::Tensor& prediction) -> void
{
    if (target.get_device() != dl::Device::GPU || prediction.get_device() != dl::Device::GPU)
    {
        throw std::runtime_error("YOLOLoss requires GPU tensors");
    }
    if (target.data() == nullptr || prediction.data() == nullptr)
    {
        throw std::runtime_error("YOLOLoss received a null device pointer");
    }
}

auto as_yolo_grid(const dl::Tensor& tensor, int final_dim, const char* name) -> dl::Tensor
{
    const std::vector<int>& shape = tensor.get_shape();
    if (shape.size() == 4)
    {
        if (shape[1] != kGridSize || shape[2] != kGridSize || shape[3] != final_dim)
        {
            throw std::runtime_error(std::string(name) + " must have shape [Batch, 7, 7, 10 + num_classes]");
        }
        return tensor.view(shape);
    }
    if (shape.size() == 2)
    {
        if (shape[1] != kCellsPerImage * final_dim)
        {
            throw std::runtime_error(std::string(name) + " must have shape [Batch, 7*7*(10 + num_classes)]");
        }
        return tensor.view({ shape[0], kGridSize, kGridSize, final_dim });
    }
    throw std::runtime_error(std::string(name) + " must be rank 2 or rank 4");
}

struct YoloWorkspace
{
    std::optional<dl::Tensor> cell_loss;
    std::optional<dl::Tensor> grad;
    std::optional<dl::Tensor> scalar;
};

auto yolo_workspace() -> YoloWorkspace&
{
    static YoloWorkspace workspace;
    return workspace;
}

} // namespace

auto YOLOLoss::calculate_iou(const dl::Tensor& box1, const dl::Tensor& box2) -> dl::Tensor
{
    require_gpu_pair(box1, box2);
    if (box1.get_size() != box2.get_size() || (box1.get_size() % static_cast<size_t>(kBoxAttrs)) != 0)
    {
        throw std::runtime_error("YOLOLoss::calculate_iou expects matching [N, 4] box tensors");
    }

    const int box_count = static_cast<int>(box1.get_size() / static_cast<size_t>(kBoxAttrs));
    dl::Tensor iou({ box_count }, dl::Device::GPU);
    if (box_count == 0)
    {
        return iou;
    }

    yolo_iou_kernel<<<launch_config(box_count), kThreads, 0, dl::current_stream()>>>(box1.data(), box2.data(), iou.data(),
        box_count);
    CHECK_CUDA(cudaGetLastError());
    return iou;
}

auto YOLOLoss::loss(const dl::Tensor& target, const dl::Tensor& prediction, int num_classes, cudaStream_t stream)
    -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("YOLOLoss_Loss");
    const dl::StreamGuard stream_guard(stream);
    if (num_classes <= 0)
    {
        throw std::runtime_error("YOLOLoss::loss requires a positive class count");
    }
    require_gpu_pair(target, prediction);

    const dl::Tensor pred_f32 = prediction.to_dtype(dl::Dtype::Float32, stream);
    const dl::Tensor tgt_f32 = target.to_dtype(dl::Dtype::Float32, stream);

    const int final_dim = 10 + num_classes;
    const dl::Tensor pred = as_yolo_grid(pred_f32, final_dim, "YOLOLoss::loss prediction");
    const dl::Tensor tgt = as_yolo_grid(tgt_f32, final_dim, "YOLOLoss::loss target");
    if (pred.get_shape()[0] != tgt.get_shape()[0])
    {
        throw std::runtime_error("YOLOLoss::loss batch sizes do not match");
    }

    const int batch_size = pred.get_shape()[0];
    const int cell_count = batch_size * kCellsPerImage;
    dl::Tensor& cell_loss = dl::Tensor::ensure(yolo_workspace().cell_loss, { cell_count }, dl::Device::GPU);

    yolo_loss_forward_kernel<<<launch_config(cell_count), kThreads, 0, stream>>>(pred.data(), tgt.data(),
        cell_loss.data(), batch_size, final_dim);
    CHECK_CUDA(cudaGetLastError());

    dl::Tensor& mean_loss = dl::Tensor::ensure(yolo_workspace().scalar, { 1 }, dl::Device::GPU);
    yolo_mean_loss_kernel<<<1, kThreads, 0, stream>>>(cell_loss.data(), mean_loss.data(), cell_count,
        dl::safe_inv(static_cast<float>(batch_size)));
    CHECK_CUDA(cudaGetLastError());
    return mean_loss.as_view();
}

auto YOLOLoss::loss_derivative(const dl::Tensor& target, const dl::Tensor& prediction, int num_classes,
    cudaStream_t stream) -> dl::Tensor
{
    const dl::NvtxRange nvtx_range("YOLOLoss_LossDerivative");
    const dl::StreamGuard stream_guard(stream);
    if (num_classes <= 0)
    {
        throw std::runtime_error("YOLOLoss::loss_derivative requires a positive class count");
    }
    require_gpu_pair(target, prediction);

    const dl::Dtype result_dtype = prediction.get_dtype();
    const dl::Tensor pred_f32 = prediction.to_dtype(dl::Dtype::Float32, stream);
    const dl::Tensor tgt_f32 = target.to_dtype(dl::Dtype::Float32, stream);

    const int final_dim = 10 + num_classes;
    const bool flattened = prediction.get_shape().size() == 2;
    const dl::Tensor pred = as_yolo_grid(pred_f32, final_dim, "YOLOLoss::loss_derivative prediction");
    const dl::Tensor tgt = as_yolo_grid(tgt_f32, final_dim, "YOLOLoss::loss_derivative target");
    if (pred.get_shape()[0] != tgt.get_shape()[0])
    {
        throw std::runtime_error("YOLOLoss::loss_derivative batch sizes do not match");
    }

    const int batch_size = pred.get_shape()[0];
    const int cell_count = batch_size * kCellsPerImage;
    dl::Tensor& grad = dl::Tensor::ensure(yolo_workspace().grad, pred.get_shape(), dl::Device::GPU);
    const float inv_batch = dl::safe_inv(static_cast<float>(batch_size));

    yolo_loss_backward_kernel<<<launch_config(cell_count), kThreads, 0, stream>>>(pred.data(), tgt.data(), grad.data(),
        batch_size, final_dim, inv_batch);
    CHECK_CUDA(cudaGetLastError());

    const float scale = dl::loss_scale();
    if (scale != 1.0F)
    {
        grad.mul_(scale);
    }
    if (flattened)
    {
        return grad.view({ batch_size, kCellsPerImage * final_dim }).to_dtype(result_dtype, stream);
    }
    return grad.to_dtype(result_dtype, stream);
}
