#pragma once

#include "DeepLearnLib/mAP.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

float calculate_iou(const cv::Rect& a, const cv::Rect& b);

std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nmsThreshold);

std::vector<Detection> decode_yolo_tensor(const std::vector<float>& output_data, float confThreshold, int imgWidth,
    int imgHeight, int numClasses);

void draw_detections(cv::Mat& img, const std::vector<Detection>& detections,
    const std::vector<std::string>& classNames,
    const cv::Scalar& defaultColor = cv::Scalar(0, 255, 0));
