#include "DeepLearnLib/TorchYOLO.hpp"

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