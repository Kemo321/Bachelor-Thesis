#include "DeepLearnLib/utils.hpp"
#include <algorithm>

float calculate_iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b;
    float inter_area = static_cast<float>(intersection.area());
    float union_area = static_cast<float>(a.area() + b.area()) - inter_area;
    return inter_area / (union_area + 1e-6f);
}

std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nms_threshold) {
    std::vector<Detection> result;
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!suppressed[j] && detections[i].class_id == detections[j].class_id) {
                if (calculate_iou(detections[i].box, detections[j].box) > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return result;
}

std::vector<Detection> decode_yolo_tensor(const torch::Tensor& output, float conf_threshold, int img_width, int img_height, int num_classes) {
    auto out_acc = output.accessor<float, 4>();
    std::vector<Detection> all_detections;

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            float max_class_prob = -1e6f; 
            int class_id = -1;
            
            for (int c = 0; c < num_classes; ++c) {
                if (out_acc[0][i][j][10 + c] > max_class_prob) { 
                    max_class_prob = out_acc[0][i][j][10 + c]; 
                    class_id = c; 
                }
            }
            
            for (int b = 0; b < 2; ++b) {
                int offset = b * 5;
                float objectness = out_acc[0][i][j][offset + 4];
                float score = objectness * max_class_prob;
                
                // Sprawdzamy czyste objectness
                if (objectness > conf_threshold) {
                    float cx = ((out_acc[0][i][j][offset + 0] + j) / 7.0f) * img_width;
                    float cy = ((out_acc[0][i][j][offset + 1] + i) / 7.0f) * img_height;
                    float w = out_acc[0][i][j][offset + 2] * img_width;
                    float h = out_acc[0][i][j][offset + 3] * img_height;
                    
                    int x_min = std::max(0, static_cast<int>(cx - w / 2));
                    int y_min = std::max(0, static_cast<int>(cy - h / 2));
                    all_detections.push_back({cv::Rect(x_min, y_min, static_cast<int>(w), static_cast<int>(h)), score, class_id});
                }
            }
        }
    }
    return all_detections;
}

void draw_detections(cv::Mat& img, const std::vector<Detection>& detections, const std::vector<std::string>& class_names, const cv::Scalar& default_color) {
    for (const auto& det : detections) {
        // Opcjonalna logika: różne kolory dla różnych klas (jak w syntetycznym zbiorze)
        cv::Scalar color = default_color;
        if (class_names.size() == 3 && class_names[0] == "square") {
            if (det.class_id == 0) color = cv::Scalar(255, 255, 255);
            else if (det.class_id == 1) color = cv::Scalar(0, 255, 0);
            else color = cv::Scalar(255, 0, 0);
        }

        cv::rectangle(img, det.box, color, 2);
        std::string label = class_names[det.class_id] + " " + std::to_string(det.score).substr(0, 4);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(cv::Point(det.box.x, det.box.y - label_size.height - 5), cv::Size(label_size.width, label_size.height + 5)), color, cv::FILLED);
        cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}