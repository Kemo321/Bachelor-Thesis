#pragma once

#include "DeepLearnLib/mAP.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

float calculate_iou(const cv::Rect& a, const cv::Rect& b);

std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nmsThreshold);

std::vector<Detection> decode_yolo_tensor(const std::vector<float>& output_data, float confThreshold, int imgWidth,
    int imgHeight, int numClasses);

/** Darknet YOLOv1 detection head: classes | objectness | boxes, with optional sqrt(w,h). */
std::vector<Detection> decode_darknet_detection(const std::vector<float>& output_data, float conf_threshold,
    int img_width, int img_height, int num_classes, int side = 7, int num_boxes = 3, bool sqrt_wh = true);

/** Darknet detection truth grid `[S,S,1+4+C]` → axis-aligned boxes for mAP. */
std::vector<Detection> detections_from_darknet_truth(const std::vector<float>& truth, int img_width, int img_height,
    int num_classes, int side = 7);

void draw_detections(cv::Mat& img, const std::vector<Detection>& detections,
    const std::vector<std::string>& classNames,
    const cv::Scalar& defaultColor = cv::Scalar(0, 255, 0));
