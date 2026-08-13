#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

/**
 * @brief Represents a single detection result.
 */
struct Detection {
    cv::Rect box; ///< Bounding box of the detection.
    float score; ///< Confidence score of the detection.
    int class_id; ///< Class ID of the detected object.
};

/**
 * @brief Calculates the Intersection over Union (IoU) between two bounding boxes.
 * 
 * @param a First bounding box.
 * @param b Second bounding box.
 * @return IoU value as a float.
 */
float calculate_iou(const cv::Rect& a, const cv::Rect& b);

/**
 * @brief Applies Non-Maximum Suppression (NMS) to filter overlapping detections.
 * 
 * @param detections Vector of detections to process.
 * @param nmsThreshold Threshold for IoU to suppress overlapping boxes.
 * @return Filtered vector of detections after applying NMS.
 */
std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nmsThreshold);

/**
 * @brief Decodes a raw YOLO tensor into a vector of detections.
 * 
 * @param output Raw tensor output from the YOLO model. Shape: [Batch, Anchors, Attributes].
 * @param confThreshold Confidence threshold for filtering detections.
 * @param imgWidth Width of the input image.
 * @param imgHeight Height of the input image.
 * @param numClasses Number of classes in the model.
 * @return Vector of decoded detections.
 */
std::vector<Detection> decode_yolo_tensor(
    const torch::Tensor& output, 
    float confThreshold, 
    int imgWidth, 
    int imgHeight, 
    int numClasses
);

/**
 * @brief Draws detection results on an image.
 * 
 * @param img Image on which to draw. Modified in-place.
 * @param detections Vector of detections to draw.
 * @param classNames Vector of class names corresponding to class IDs.
 * @param defaultColor Default color for drawing bounding boxes (optional).
 */
void draw_detections(
    cv::Mat& img, 
    const std::vector<Detection>& detections, 
    const std::vector<std::string>& classNames,
    const cv::Scalar& defaultColor = cv::Scalar(0, 255, 0)
);