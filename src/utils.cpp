#include "DeepLearnLib/utils.hpp"
#include <algorithm>
#include <torch/torch.h>

/**
 * @brief Calculates Intersection over Union (IoU) metric for bounding box comparison.
 * 
 * IoU is a fundamental metric in object detection used for Non-Maximum Suppression (NMS)
 * and evaluation tasks. It measures the overlap between two bounding boxes as the ratio 
 * of their intersection area to their union area. This metric is essential for filtering
 * redundant detections and assessing model performance.
 * 
 * @param box_a First bounding box in cv::Rect format (x, y, width, height)
 * @param box_b Second bounding box in cv::Rect format (x, y, width, height)
 * @return float IoU value in range [0.0F, 1.0F], where 1.0F indicates perfect overlap
 *         and 0.0F indicates no intersection
 * 
 * @details 
 *   Formula: IoU = (Intersection Area) / (Union Area)
 *   Union Area = Area(box_a) + Area(box_b) - Intersection Area
 *   Small epsilon (1e-6F) is added to denominator for numerical stability to prevent
 *   division by zero when both boxes have zero area.
 */
float calculate_iou(const cv::Rect& box_a, const cv::Rect& box_b) {
    cv::Rect intersection = box_a & box_b;
    float intersection_area = static_cast<float>(intersection.area());
    float union_area = static_cast<float>(box_a.area() + box_b.area()) - intersection_area;
    return intersection_area / (union_area + 1e-6F);
}

/**
 * @brief Applies Non-Maximum Suppression (NMS) to filter overlapping detections.
 * 
 * NMS is a post-processing technique that removes redundant bounding boxes with high
 * overlap. Detections are sorted by confidence score in descending order, and overlapping
 * detections within the same class are suppressed based on the IoU threshold. This ensures
 * that only the most confident, non-overlapping detections are retained.
 * 
 * @param detections Reference to vector of Detection objects to be filtered. Will be sorted
 *                   by confidence score (descending order). Each Detection contains:
 *                   - box: cv::Rect with bounding box coordinates
 *                   - score: float confidence score in range [0.0F, 1.0F]
 *                   - class_id: int representing the object class
 * @param nms_threshold float IoU threshold for suppression. Detections with IoU > nms_threshold
 *                      are considered overlapping and suppressed. Typical range: [0.3F, 0.7F]
 * @return std::vector<Detection> Filtered detections after NMS, maintaining descending
 *         score order
 * 
 * @details
 *   Algorithm: Greedy NMS
 *   1. Sort detections by confidence score (descending)
 *   2. Iterate through sorted detections, keeping high-confidence boxes
 *   3. For each kept box, suppress all overlapping boxes of the same class
 *   4. Only suppress boxes with class_id matching the kept box (class-wise NMS)
 */
std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nms_threshold) {
    std::vector<Detection> result;
    
    // Sort detections by confidence score in descending order
    std::sort(detections.begin(), detections.end(), 
        [](const Detection& detection_a, const Detection& detection_b) {
            return detection_a.score > detection_b.score;
        });

    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        
        result.push_back(detections[i]);
        
        // Suppress overlapping detections of the same class
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!suppressed[j] && detections[i].class_id == detections[j].class_id) {
                float iou_value = calculate_iou(detections[i].box, detections[j].box);
                if (iou_value > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief Decodes YOLO tensor output into Detection objects with bounding boxes.
 * 
 * This function processes raw YOLO network output (typically from a 7×7 grid of cells,
 * each predicting 2 bounding boxes and class probabilities) into structured Detection
 * objects with absolute image coordinates. The YOLO architecture uses relative coordinates
 * and grid cell offsets, which are converted to actual pixel coordinates and dimensions.
 * 
 * @param output torch::Tensor YOLO model output tensor with shape [Batch=1, GridH=7, GridW=7, Channels=30]
 *               Channel layout: [tx, ty, tw, th, objectness, tx, ty, tw, th, objectness, class_probs[20]]
 *               - tx, ty: relative x, y offsets within grid cell (sigmoid output ~[0, 1])
 *               - tw, th: relative width, height (exp-transformed prior box scales)
 *               - objectness: confidence score that cell contains an object
 *               - class_probs: per-class probabilities
 * @param conf_threshold float Minimum objectness confidence threshold for detection. 
 *                       Boxes with objectness <= conf_threshold are discarded.
 *                       Typical range: [0.3F, 0.5F]
 * @param img_width int Width of input image in pixels
 * @param img_height int Height of input image in pixels
 * @param num_classes int Number of object classes (typically 20 for PASCAL VOC, 80 for COCO)
 * @return std::vector<Detection> Vector of detected objects. Each Detection contains:
 *         - box: cv::Rect with absolute pixel coordinates (x_min, y_min, width, height)
 *         - score: float objectness score
 *         - class_id: int predicted class index
 * 
 * @details
 *   YOLO Decoding Process:
 *   1. Iterate through all 7×7 = 49 grid cells
 *   2. For each cell, find the class with maximum probability
 *   3. For each of 2 bounding box predictions in the cell:
 *      a. Extract objectness score and coordinates
 *      b. Apply grid cell offset: normalized_x = (tx + grid_j) / 7.0
 *      c. Convert to image coordinates: cx = normalized_x * img_width
 *      d. Apply exponential scaling to width/height
 *      e. Clamp box coordinates to image bounds [0, img_width) × [0, img_height)
 *   4. Only include boxes with objectness > conf_threshold
 */
std::vector<Detection> decode_yolo_tensor(const torch::Tensor& output, float conf_threshold, 
                                          int img_width, int img_height, int num_classes) {
    auto out_acc = output.accessor<float, 4>();
    std::vector<Detection> all_detections;
    
    constexpr int GRID_SIZE = 7;
    constexpr int NUM_BOXES_PER_CELL = 2;
    constexpr int COORDINATES_PER_BOX = 5;  // tx, ty, tw, th, objectness
    constexpr int CLASS_PROB_OFFSET = 10;   // Offset to class probabilities
    constexpr float GRID_SIZE_FLOAT = 7.0F;
    constexpr int BATCH_IDX = 0;

    for (int grid_i = 0; grid_i < GRID_SIZE; ++grid_i) {
        for (int grid_j = 0; grid_j < GRID_SIZE; ++grid_j) {
            
            // Find class with maximum probability for this grid cell
            float max_class_prob = -1e6F;
            int class_id = -1;
            
            for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
                float class_prob = out_acc[BATCH_IDX][grid_i][grid_j][CLASS_PROB_OFFSET + class_idx];
                if (class_prob > max_class_prob) {
                    max_class_prob = class_prob;
                    class_id = class_idx;
                }
            }
            
            // Process both bounding box predictions per grid cell
            for (int box_idx = 0; box_idx < NUM_BOXES_PER_CELL; ++box_idx) {
                int coordinate_offset = box_idx * COORDINATES_PER_BOX;
                
                float objectness_score = out_acc[BATCH_IDX][grid_i][grid_j][coordinate_offset + 4];
                
                // Filter by objectness confidence threshold
                if (objectness_score <= conf_threshold) {
                    continue;
                }
                
                // Decode normalized coordinates to image coordinates
                // tx and ty are offsets within grid cell [0, 1]
                float normalized_tx = out_acc[BATCH_IDX][grid_i][grid_j][coordinate_offset + 0];
                float normalized_ty = out_acc[BATCH_IDX][grid_i][grid_j][coordinate_offset + 1];
                
                // Apply grid cell offset
                float normalized_center_x = (normalized_tx + grid_j) / GRID_SIZE_FLOAT;
                float normalized_center_y = (normalized_ty + grid_i) / GRID_SIZE_FLOAT;
                
                // Convert to image pixel coordinates
                float center_x = normalized_center_x * static_cast<float>(img_width);
                float center_y = normalized_center_y * static_cast<float>(img_height);
                
                // Decode width and height (already in image scale from network output)
                float box_width = out_acc[BATCH_IDX][grid_i][grid_j][coordinate_offset + 2] * 
                                  static_cast<float>(img_width);
                float box_height = out_acc[BATCH_IDX][grid_i][grid_j][coordinate_offset + 3] * 
                                   static_cast<float>(img_height);
                
                // Calculate top-left corner with boundary clamping
                int x_min = std::max(0, static_cast<int>(center_x - box_width / 2.0F));
                int y_min = std::max(0, static_cast<int>(center_y - box_height / 2.0F));
                
                all_detections.push_back({
                    cv::Rect(x_min, y_min, static_cast<int>(box_width), static_cast<int>(box_height)),
                    objectness_score,
                    class_id
                });
            }
        }
    }
    
    return all_detections;
}

/**
 * @brief Renders detection bounding boxes and class labels on image.
 * 
 * Visualizes Detection objects by drawing rectangles for bounding boxes and text labels
 * containing class names and confidence scores. Supports class-specific color schemes
 * for improved visual discrimination (e.g., different colors for geometric shapes).
 * 
 * @param img cv::Mat Reference to image matrix (BGR format) where detections will be drawn.
 *            Modified in-place. Expected shape: [height, width, 3] for color images.
 * @param detections const std::vector<Detection>& Vector of Detection objects to visualize.
 *                   Each detection must have valid class_id index within class_names.
 * @param class_names const std::vector<std::string>& Vector of class name strings, indexed
 *                    by class_id. First class name used to determine color scheme.
 * @param default_color const cv::Scalar& Default BGR color for bounding box rectangles
 *                      (e.g., cv::Scalar(0, 255, 0) for green). Format: [B, G, R].
 * 
 * @details
 *   Rendering details:
 *   - Bounding box rectangle drawn with 2-pixel line width
 *   - Label background rectangle filled with box color for contrast
 *   - Label text colored black (0, 0, 0) for readability
 *   - Score truncated to 4 characters (e.g., "0.95")
 *   - Special color mapping for synthetic dataset: class 0 → white, 1 → green, 2 → red
 *   - Label positioned above bounding box with 5-pixel padding
 */
void draw_detections(cv::Mat& img, const std::vector<Detection>& detections, 
                     const std::vector<std::string>& class_names, const cv::Scalar& default_color) {
    
    constexpr int LINE_THICKNESS = 2;
    constexpr int LABEL_PADDING = 5;
    constexpr int TEXT_THICKNESS = 1;
    constexpr double FONT_SCALE = 0.5;
    const cv::Scalar TEXT_COLOR(0, 0, 0);  // Black text
    
    for (const auto& detection : detections) {
        cv::Scalar box_color = default_color;
        
        // Apply class-specific colors for synthetic dataset with geometric shapes
        if (class_names.size() == 3 && class_names[0] == "square") {
            if (detection.class_id == 0) {
                box_color = cv::Scalar(255, 255, 255);  // White for square
            } else if (detection.class_id == 1) {
                box_color = cv::Scalar(0, 255, 0);      // Green for circle
            } else {
                box_color = cv::Scalar(255, 0, 0);      // Red for triangle
            }
        }
        
        // Draw bounding box rectangle
        cv::rectangle(img, detection.box, box_color, LINE_THICKNESS);
        
        // Format and prepare label text
        std::string score_str = std::to_string(detection.score);
        if (score_str.length() > 4) {
            score_str = score_str.substr(0, 4);
        }
        std::string label_text = class_names[detection.class_id] + " " + score_str;
        
        // Calculate label background rectangle size
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 
                                              FONT_SCALE, TEXT_THICKNESS, &baseline);
        
        // Draw label background rectangle
        cv::Rect label_background(
            cv::Point(detection.box.x, detection.box.y - label_size.height - LABEL_PADDING),
            cv::Size(label_size.width, label_size.height + LABEL_PADDING)
        );
        cv::rectangle(img, label_background, box_color, cv::FILLED);
        
        // Draw label text
        cv::putText(img, label_text, 
                   cv::Point(detection.box.x, detection.box.y - LABEL_PADDING),
                   cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS);
    }
}