#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// 1. Struktura przechowująca detekcję
struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

// 2. Operacje na ramkach
float calculate_iou(const cv::Rect& a, const cv::Rect& b);
std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nms_threshold);

// 3. Dekodowanie surowego tensora YOLO na wektor detekcji
std::vector<Detection> decode_yolo_tensor(
    const torch::Tensor& output, 
    float conf_threshold, 
    int img_width, 
    int img_height, 
    int num_classes
);

// 4. Uniwersalne rysowanie po obrazie (z obsługą wielu kolorów)
void draw_detections(
    cv::Mat& img, 
    const std::vector<Detection>& detections, 
    const std::vector<std::string>& class_names,
    const cv::Scalar& default_color = cv::Scalar(0, 255, 0)
);