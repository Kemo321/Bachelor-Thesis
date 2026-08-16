#include "DeepLearnLib/mAP.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

namespace
{

constexpr float kIouEps = 1e-6F;
constexpr int kVocPoints = 11;

auto box_area(const Detection& detection) -> float
{
    return std::max(0.0F, detection.width) * std::max(0.0F, detection.height);
}

auto average_precision_for_class(const std::vector<Detection>& predicted, const std::vector<Detection>& ground_truth,
    int class_id, float iou_threshold) -> float
{
    std::vector<Detection> class_pred;
    std::vector<Detection> class_gt;
    class_pred.reserve(predicted.size());
    class_gt.reserve(ground_truth.size());
    for (const auto& detection : predicted)
    {
        if (detection.class_id == class_id)
        {
            class_pred.push_back(detection);
        }
    }
    for (const auto& detection : ground_truth)
    {
        if (detection.class_id == class_id)
        {
            class_gt.push_back(detection);
        }
    }

    if (class_gt.empty())
    {
        return class_pred.empty() ? 0.0F : 0.0F;
    }

    std::sort(class_pred.begin(), class_pred.end(),
        [](const Detection& lhs, const Detection& rhs)
        {
            return lhs.score > rhs.score;
        });

    std::vector<char> matched(class_gt.size(), 0);
    std::vector<float> true_positive(class_pred.size(), 0.0F);
    std::vector<float> false_positive(class_pred.size(), 0.0F);

    for (std::size_t pred_index = 0; pred_index < class_pred.size(); ++pred_index)
    {
        float best_iou = 0.0F;
        int best_gt = -1;
        for (std::size_t gt_index = 0; gt_index < class_gt.size(); ++gt_index)
        {
            if (matched[gt_index] != 0)
            {
                continue;
            }
            const float iou = detection_iou(class_pred[pred_index], class_gt[gt_index]);
            if (iou > best_iou)
            {
                best_iou = iou;
                best_gt = static_cast<int>(gt_index);
            }
        }

        if (best_gt >= 0 && best_iou >= iou_threshold)
        {
            matched[static_cast<std::size_t>(best_gt)] = 1;
            true_positive[pred_index] = 1.0F;
        }
        else
        {
            false_positive[pred_index] = 1.0F;
        }
    }

    std::vector<float> precision(class_pred.size(), 0.0F);
    std::vector<float> recall(class_pred.size(), 0.0F);
    float running_tp = 0.0F;
    float running_fp = 0.0F;
    const float gt_count = static_cast<float>(class_gt.size());
    for (std::size_t index = 0; index < class_pred.size(); ++index)
    {
        running_tp += true_positive[index];
        running_fp += false_positive[index];
        precision[index] = running_tp / std::max(running_tp + running_fp, kIouEps);
        recall[index] = running_tp / gt_count;
    }

    float average_precision = 0.0F;
    for (int point = 0; point < kVocPoints; ++point)
    {
        const float recall_threshold = static_cast<float>(point) / static_cast<float>(kVocPoints - 1);
        float max_precision = 0.0F;
        for (std::size_t index = 0; index < precision.size(); ++index)
        {
            if (recall[index] >= recall_threshold)
            {
                max_precision = std::max(max_precision, precision[index]);
            }
        }
        if (class_pred.empty() && recall_threshold == 0.0F)
        {
            max_precision = 0.0F;
        }
        average_precision += max_precision;
    }
    return average_precision / static_cast<float>(kVocPoints);
}

} // namespace

auto detection_iou(const Detection& predicted, const Detection& ground_truth) -> float
{
    const float left = std::max(predicted.x, ground_truth.x);
    const float top = std::max(predicted.y, ground_truth.y);
    const float right = std::min(predicted.x + predicted.width, ground_truth.x + ground_truth.width);
    const float bottom = std::min(predicted.y + predicted.height, ground_truth.y + ground_truth.height);
    const float intersection_w = std::max(0.0F, right - left);
    const float intersection_h = std::max(0.0F, bottom - top);
    const float intersection = intersection_w * intersection_h;
    const float union_area = box_area(predicted) + box_area(ground_truth) - intersection;
    return intersection / (union_area + kIouEps);
}

auto mean_average_precision(const std::vector<Detection>& predicted, const std::vector<Detection>& ground_truth,
    float iou_threshold) -> float
{
    if (iou_threshold < 0.0F || iou_threshold > 1.0F)
    {
        throw std::runtime_error("mean_average_precision IoU threshold must be in [0, 1]");
    }
    if (ground_truth.empty())
    {
        return 0.0F;
    }

    std::set<int> classes;
    for (const auto& detection : ground_truth)
    {
        classes.insert(detection.class_id);
    }
    for (const auto& detection : predicted)
    {
        classes.insert(detection.class_id);
    }

    float sum_ap = 0.0F;
    int counted = 0;
    for (int class_id : classes)
    {
        bool has_gt = false;
        for (const auto& detection : ground_truth)
        {
            if (detection.class_id == class_id)
            {
                has_gt = true;
                break;
            }
        }
        if (!has_gt)
        {
            continue;
        }
        sum_ap += average_precision_for_class(predicted, ground_truth, class_id, iou_threshold);
        ++counted;
    }
    if (counted == 0)
    {
        return 0.0F;
    }
    return sum_ap / static_cast<float>(counted);
}
