#include "DeepLearnLib/utils.hpp"
#include "DeepLearnLib/SafeMath.hpp"

#include <algorithm>
#include <execution>
#include <numeric>
#include <stdexcept>
#include <vector>

float calculate_iou(const cv::Rect& box_a, const cv::Rect& box_b)
{
    cv::Rect intersection = box_a & box_b;
    float intersection_area = static_cast<float>(intersection.area());
    float union_area = static_cast<float>(box_a.area() + box_b.area()) - intersection_area;
    return dl::guarded_div(intersection_area, union_area);
}

std::vector<Detection> apply_nms(std::vector<Detection>& detections, float nms_threshold)
{
    std::sort(std::execution::par_unseq, detections.begin(), detections.end(),
        [](const Detection& detection_a, const Detection& detection_b)
        {
            return detection_a.score > detection_b.score;
        });

    const std::size_t count = detections.size();
    std::vector<char> suppressed(count, 0);
    std::vector<float> pairwise_iou(count * count, 0.0F);
    std::vector<std::size_t> rows(count);
    std::iota(rows.begin(), rows.end(), 0);

    std::for_each(std::execution::par_unseq, rows.begin(), rows.end(),
        [&](std::size_t i)
        {
            for (std::size_t j = i + 1; j < count; ++j)
            {
                if (detections[i].class_id != detections[j].class_id)
                {
                    continue;
                }
                const cv::Rect box_i(static_cast<int>(detections[i].x), static_cast<int>(detections[i].y),
                    static_cast<int>(detections[i].width), static_cast<int>(detections[i].height));
                const cv::Rect box_j(static_cast<int>(detections[j].x), static_cast<int>(detections[j].y),
                    static_cast<int>(detections[j].width), static_cast<int>(detections[j].height));
                pairwise_iou[(i * count) + j] = calculate_iou(box_i, box_j);
            }
        });

    std::vector<Detection> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        if (suppressed[i] != 0)
        {
            continue;
        }

        result.push_back(detections[i]);
        for (std::size_t j = i + 1; j < count; ++j)
        {
            if (suppressed[j] == 0 && pairwise_iou[(i * count) + j] > nms_threshold)
            {
                suppressed[j] = 1;
            }
        }
    }

    return result;
}

std::vector<Detection> decode_yolo_tensor(const std::vector<float>& output_data, float conf_threshold,
    int img_width, int img_height, int num_classes)
{
    constexpr int GRID_SIZE = 7;
    constexpr int NUM_BOXES_PER_CELL = 2;
    constexpr int COORDINATES_PER_BOX = 5; // tx, ty, tw, th, objectness
    constexpr int CLASS_PROB_OFFSET = 10;
    constexpr float GRID_SIZE_FLOAT = 7.0F;
    constexpr int CELL_COUNT = GRID_SIZE * GRID_SIZE;
    const int attributes = 10 + num_classes;
    const size_t expected_size = static_cast<size_t>(CELL_COUNT * attributes);
    if (output_data.size() < expected_size)
    {
        throw std::runtime_error("decode_yolo_tensor expected a flat [1, 7, 7, 10+num_classes] buffer");
    }

    auto at = [&](int grid_i, int grid_j, int offset) -> float
    {
        return output_data[static_cast<size_t>((grid_i * GRID_SIZE * attributes) + (grid_j * attributes) + offset)];
    };

    const std::size_t slot_count = static_cast<std::size_t>(CELL_COUNT * NUM_BOXES_PER_CELL);
    std::vector<Detection> slots(slot_count);
    std::vector<char> valid(slot_count, 0);
    std::vector<int> cells(CELL_COUNT);
    std::iota(cells.begin(), cells.end(), 0);

    std::for_each(std::execution::par_unseq, cells.begin(), cells.end(),
        [&](int cell)
        {
            const int grid_i = cell / GRID_SIZE;
            const int grid_j = cell % GRID_SIZE;

            float max_class_prob = -1e6F;
            int class_id = -1;
            for (int class_idx = 0; class_idx < num_classes; ++class_idx)
            {
                const float class_prob = at(grid_i, grid_j, CLASS_PROB_OFFSET + class_idx);
                if (class_prob > max_class_prob)
                {
                    max_class_prob = class_prob;
                    class_id = class_idx;
                }
            }

            for (int box_idx = 0; box_idx < NUM_BOXES_PER_CELL; ++box_idx)
            {
                const int coordinate_offset = box_idx * COORDINATES_PER_BOX;
                const float objectness_score = at(grid_i, grid_j, coordinate_offset + 4);
                if (objectness_score <= conf_threshold)
                {
                    continue;
                }

                const float normalized_tx = at(grid_i, grid_j, coordinate_offset + 0);
                const float normalized_ty = at(grid_i, grid_j, coordinate_offset + 1);
                const float normalized_center_x = (normalized_tx + static_cast<float>(grid_j)) / GRID_SIZE_FLOAT;
                const float normalized_center_y = (normalized_ty + static_cast<float>(grid_i)) / GRID_SIZE_FLOAT;
                const float center_x = normalized_center_x * static_cast<float>(img_width);
                const float center_y = normalized_center_y * static_cast<float>(img_height);
                const float box_width = at(grid_i, grid_j, coordinate_offset + 2) * static_cast<float>(img_width);
                const float box_height = at(grid_i, grid_j, coordinate_offset + 3) * static_cast<float>(img_height);

                const int x_min = std::max(0, static_cast<int>(center_x - box_width / 2.0F));
                const int y_min = std::max(0, static_cast<int>(center_y - box_height / 2.0F));

                const std::size_t slot = (static_cast<std::size_t>(cell) * NUM_BOXES_PER_CELL)
                    + static_cast<std::size_t>(box_idx);
                slots[slot] = Detection { static_cast<float>(x_min), static_cast<float>(y_min), box_width, box_height,
                    objectness_score, class_id };
                valid[slot] = 1;
            }
        });

    std::vector<Detection> all_detections;
    all_detections.reserve(slot_count);
    for (std::size_t slot = 0; slot < slot_count; ++slot)
    {
        if (valid[slot] != 0)
        {
            all_detections.push_back(slots[slot]);
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
