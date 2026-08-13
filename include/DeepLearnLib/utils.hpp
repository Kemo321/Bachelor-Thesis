#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief Represents a single detection result.
 */
struct Detection
{
    cv::Rect box; ///< Bounding box of the detection.
    float score; ///< Confidence score of the detection.
    int class_id; ///< Class ID of the detected object.
};

/**
 * @brief Calculates the Intersection over Union (IoU) between two bounding boxes.
 */
float calculate_iou(const cv::Rect& a, const cv::Rect& b);

/**
 * @brief Applies Non-Maximum Suppression (NMS) to filter overlapping detections.
 */
std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nmsThreshold);

/**
 * @brief Decodes a raw YOLOv1 output buffer into detections.
 *
 * @param output_data Flat row-major buffer with layout [Batch=1, Grid=7, Grid=7, Attributes=10+numClasses].
 * @param confThreshold Confidence threshold for filtering detections.
 * @param imgWidth Width of the original image.
 * @param imgHeight Height of the original image.
 * @param numClasses Number of classes in the model.
 */
std::vector<Detection> decode_yolo_tensor(const std::vector<float>& output_data, float confThreshold, int imgWidth,
                                          int imgHeight, int numClasses);

/**
 * @brief Draws detection results on an image.
 */
void draw_detections(cv::Mat& img, const std::vector<Detection>& detections,
                     const std::vector<std::string>& classNames,
                     const cv::Scalar& defaultColor = cv::Scalar(0, 255, 0));
