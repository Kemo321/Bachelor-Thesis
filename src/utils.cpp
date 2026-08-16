#include "DeepLearnLib/utils.hpp"
#include <algorithm>
#include <stdexcept>

float calculate_iou(const cv::Rect& box_a, const cv::Rect& box_b)
{
    cv::Rect intersection = box_a & box_b;
    float intersection_area = static_cast<float>(intersection.area());
    float union_area = static_cast<float>(box_a.area() + box_b.area()) - intersection_area;
    return intersection_area / (union_area + 1e-6F);
}

std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nms_threshold)
{
    std::vector<Detection> result;

    std::sort(detections.begin(), detections.end(),
        [](const Detection& detection_a, const Detection& detection_b)
        {
            return detection_a.score > detection_b.score;
        });

    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppressed[i])
        {
            continue;
        }

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (!suppressed[j] && detections[i].class_id == detections[j].class_id)
            {
                const cv::Rect box_i(static_cast<int>(detections[i].x), static_cast<int>(detections[i].y),
                    static_cast<int>(detections[i].width), static_cast<int>(detections[i].height));
                const cv::Rect box_j(static_cast<int>(detections[j].x), static_cast<int>(detections[j].y),
                    static_cast<int>(detections[j].width), static_cast<int>(detections[j].height));
                float iou_value = calculate_iou(box_i, box_j);
                if (iou_value > nms_threshold)
                {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

std::vector<Detection> decode_yolo_tensor(const std::vector<float>& output_data, float conf_threshold,
    int img_width, int img_height, int num_classes)
{
    std::vector<Detection> all_detections;

    constexpr int GRID_SIZE = 7;
    constexpr int NUM_BOXES_PER_CELL = 2;
    constexpr int COORDINATES_PER_BOX = 5; // tx, ty, tw, th, objectness
    constexpr int CLASS_PROB_OFFSET = 10;
    constexpr float GRID_SIZE_FLOAT = 7.0F;
    const int attributes = 10 + num_classes;
    const size_t expected_size = static_cast<size_t>(GRID_SIZE * GRID_SIZE * attributes);
    if (output_data.size() < expected_size)
    {
        throw std::runtime_error("decode_yolo_tensor expected a flat [1, 7, 7, 10+num_classes] buffer");
    }

    auto at = [&](int grid_i, int grid_j, int offset) -> float
    {
        return output_data[static_cast<size_t>((grid_i * GRID_SIZE * attributes) + (grid_j * attributes) + offset)];
    };

    for (int grid_i = 0; grid_i < GRID_SIZE; ++grid_i)
    {
        for (int grid_j = 0; grid_j < GRID_SIZE; ++grid_j)
        {
            float max_class_prob = -1e6F;
            int class_id = -1;

            for (int class_idx = 0; class_idx < num_classes; ++class_idx)
            {
                float class_prob = at(grid_i, grid_j, CLASS_PROB_OFFSET + class_idx);
                if (class_prob > max_class_prob)
                {
                    max_class_prob = class_prob;
                    class_id = class_idx;
                }
            }

            for (int box_idx = 0; box_idx < NUM_BOXES_PER_CELL; ++box_idx)
            {
                int coordinate_offset = box_idx * COORDINATES_PER_BOX;

                float objectness_score = at(grid_i, grid_j, coordinate_offset + 4);
                if (objectness_score <= conf_threshold)
                {
                    continue;
                }

                float normalized_tx = at(grid_i, grid_j, coordinate_offset + 0);
                float normalized_ty = at(grid_i, grid_j, coordinate_offset + 1);
                float normalized_center_x = (normalized_tx + static_cast<float>(grid_j)) / GRID_SIZE_FLOAT;
                float normalized_center_y = (normalized_ty + static_cast<float>(grid_i)) / GRID_SIZE_FLOAT;
                float center_x = normalized_center_x * static_cast<float>(img_width);
                float center_y = normalized_center_y * static_cast<float>(img_height);
                float box_width = at(grid_i, grid_j, coordinate_offset + 2) * static_cast<float>(img_width);
                float box_height = at(grid_i, grid_j, coordinate_offset + 3) * static_cast<float>(img_height);

                int x_min = std::max(0, static_cast<int>(center_x - box_width / 2.0F));
                int y_min = std::max(0, static_cast<int>(center_y - box_height / 2.0F));

                all_detections.push_back({ static_cast<float>(x_min), static_cast<float>(y_min), box_width, box_height,
                    objectness_score, class_id });
            }
        }
    }

    return all_detections;
}

void draw_detections(cv::Mat& img, const std::vector<Detection>& detections,
    const std::vector<std::string>& class_names, const cv::Scalar& default_color)
{
    constexpr int LINE_THICKNESS = 2;
    constexpr int LABEL_PADDING = 5;
    constexpr int TEXT_THICKNESS = 1;
    constexpr double FONT_SCALE = 0.5;
    const cv::Scalar TEXT_COLOR(0, 0, 0);

    for (const auto& detection : detections)
    {
        cv::Scalar box_color = default_color;

        if (class_names.size() == 3 && class_names[0] == "square")
        {
            if (detection.class_id == 0)
            {
                box_color = cv::Scalar(255, 255, 255);
            }
            else if (detection.class_id == 1)
            {
                box_color = cv::Scalar(0, 255, 0);
            }
            else
            {
                box_color = cv::Scalar(255, 0, 0);
            }
        }

        const cv::Rect box(static_cast<int>(detection.x), static_cast<int>(detection.y),
            static_cast<int>(detection.width), static_cast<int>(detection.height));

        cv::rectangle(img, box, box_color, LINE_THICKNESS);

        std::string score_str = std::to_string(detection.score);
        if (score_str.length() > 4)
        {
            score_str = score_str.substr(0, 4);
        }
        std::string label_text = class_names[detection.class_id] + " " + score_str;

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, TEXT_THICKNESS, &baseline);

        cv::Rect label_background(
            cv::Point(box.x, box.y - label_size.height - LABEL_PADDING),
            cv::Size(label_size.width, label_size.height + LABEL_PADDING));
        cv::rectangle(img, label_background, box_color, cv::FILLED);

        cv::putText(img, label_text,
            cv::Point(box.x, box.y - LABEL_PADDING),
            cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS);
    }
}
