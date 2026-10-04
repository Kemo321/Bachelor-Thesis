#pragma once

#include "DeepLearnLib/mAP.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

float calculate_iou(const cv::Rect& a, const cv::Rect& b);

std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nms_threshold);

std::vector<Detection> decode_yolo_tensor(const std::vector<float>& output_data, float conf_threshold, int img_width,
    int img_height, int num_classes);

void draw_detections(cv::Mat& img, const std::vector<Detection>& detections,
    const std::vector<std::string>& class_names, const cv::Scalar& default_color = cv::Scalar(0, 255, 0));
