#include "TorchYOLO.hpp"

/**
 * @brief Constructor for YOLOv1 neural network implementation.
 *
 * Initializes the YOLOv1 backbone (feature extraction) and head (detection output) modules.
 * The backbone implements a modified AlexNet architecture with depthwise feature extraction,
 * while the head performs spatial regression of bounding boxes and class predictions.
 *
 * @param num_classes Number of object classes for detection (e.g., 20 for PASCAL VOC).
 *
 * Architecture overview:
 * - Backbone: 24 convolutional layers + batch normalization + LeakyReLU activation
 * - Head: 2 fully connected layers for flattened spatial features to detection grid
 * - Final output shape: [Batch, 7, 7, (10 + num_classes)]
 *   where 10 = 2 bounding boxes × (4 coords + 1 confidence)
 */
YOLOv1Impl::YOLOv1Impl(int num_classes) : num_classes_(num_classes)
{
    using namespace torch::nn;

    // Feature extraction backbone: progressively increases receptive field while reducing spatial dimensions
    backbone = register_module("backbone", Sequential(
        // Initial large kernel convolution with stride for rapid downsampling
        Conv2d(Conv2dOptions(3, 64, 7).stride(2).padding(3)),
        BatchNorm2d(64),
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        // Second conv block: channel expansion from 64 to 192
        Conv2d(Conv2dOptions(64, 192, 3).padding(1)),
        BatchNorm2d(192),
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        // Multi-scale feature extraction blocks with 1×1 bottleneck convolutions
        // 1×1 convolutions reduce dimensionality before expensive 3×3 operations
        Conv2d(Conv2dOptions(192, 128, 1)), BatchNorm2d(128), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(128, 256, 3).padding(1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(256, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        // Intermediate feature blocks: channel dimension increases to 512
        Conv2d(Conv2dOptions(512, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(512, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(512, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(512, 256, 1)), BatchNorm2d(256), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(256, 512, 3).padding(1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(512, 512, 1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        // Final deep feature extraction: channel expansion to 1024
        Conv2d(Conv2dOptions(1024, 512, 1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(1024, 512, 1)), BatchNorm2d(512), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(512, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),

        // Detection-specific layers: stride=2 reduces spatial dims from 14×14 to 7×7 (YOLO grid)
        Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(1024, 1024, 3).stride(2).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Conv2d(Conv2dOptions(1024, 1024, 3).padding(1)), BatchNorm2d(1024), LeakyReLU(LeakyReLUOptions().negative_slope(0.1F))
    ));

    // Detection head: fully connected layers mapping spatial features to detection predictions
    // Input: [Batch, 1024*7*7] flattened features
    // Output: [Batch, 7*7*(10 + num_classes)] detection grid predictions
    head = register_module("head", Sequential(
        Linear(1024 * 7 * 7, 4096),
        LeakyReLU(LeakyReLUOptions().negative_slope(0.1F)),
        Dropout(0.5),
        Linear(4096, 7 * 7 * (10 + num_classes))
    ));
}

/**
 * @brief Forward pass through YOLOv1 network.
 *
 * Processes input images through backbone feature extraction and detection head,
 * producing spatial predictions for bounding boxes and class probabilities.
 *
 * @param input_tensor Input tensor of shape [Batch, 3, 448, 448] (RGB images normalized to [-1, 1]).
 *
 * @return Output tensor of shape [Batch, 7, 7, (10 + num_classes)] containing:
 *         - (x, y, w, h, confidence) × 2 bounding boxes per grid cell (5*2=10 values)
 *         - Class probability distribution (num_classes values)
 *         Note: x, y, w, h are normalized coordinates relative to grid cell and image dimensions.
 */
auto YOLOv1Impl::forward(torch::Tensor input_tensor) -> torch::Tensor
{
    // Extract spatial features via backbone: [Batch, 3, 448, 448] → [Batch, 1024, 7, 7]
    auto features = backbone->forward(input_tensor);

    // Flatten spatial dimensions for fully connected layers
    // [Batch, 1024, 7, 7] → [Batch, 1024*7*7]
    // Contiguous ensures memory layout is compatible with linear layer operations
    auto flattened = features.view({ features.size(0), -1 }).contiguous();

    // Apply fully connected detection head
    // [Batch, 1024*7*7] → [Batch, 7*7*(10 + num_classes)]
    auto predictions = head->forward(flattened);

    // Reshape to grid format for spatial localization
    // [Batch, 7*7*(10 + num_classes)] → [Batch, 7, 7, (10 + num_classes)]
    // Contiguous ensures proper memory layout for downstream processing and loss computation
    return predictions.view({ -1, 7, 7, 10 + num_classes_ }).contiguous();
}

auto compute_yolo_loss(const torch::Tensor& prediction, const torch::Tensor& target) -> torch::Tensor
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
    const auto iou1 = box_iou((p1x + col) / grid, (p1y + row) / grid, p1w, p1h, (tx + col) / grid, (ty + row) / grid, tw, th);
    const auto iou2 = box_iou((p2x + col) / grid, (p2y + row) / grid, p2w, p2h, (tx + col) / grid, (ty + row) / grid, tw, th);

    const auto box2_better = (iou2 > iou1).to(pred.dtype());
    const auto resp_b1 = (1.0 - box2_better) * obj;
    const auto resp_b2 = box2_better * obj;
    const auto noobj_b1 = 1.0 - resp_b1;
    const auto noobj_b2 = 1.0 - resp_b2;

    auto sqrt_safe = [](const torch::Tensor& value) { return torch::sqrt(value.clamp_min(kEps)); };
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
