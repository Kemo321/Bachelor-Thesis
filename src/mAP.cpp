#include "DeepLearnLib/mAP.hpp"
#include "DeepLearnLib/SafeMath.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <functional>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace
{

constexpr int kVocPoints = 11;

auto box_area(const Detection& detection) -> float
{
    return std::max(0.0F, detection.width) * std::max(0.0F, detection.height);
}

auto average_precision_for_class(const std::vector<Detection>& predicted, const std::vector<Detection>& ground_truth,
    int class_id, float iou_threshold) -> float
{
    std::vector<Detection> class_pred(predicted.size());
    std::vector<Detection> class_gt(ground_truth.size());
    const auto pred_end = std::copy_if(std::execution::par_unseq, predicted.begin(), predicted.end(), class_pred.begin(),
        [class_id](const Detection& detection)
        {
            return detection.class_id == class_id;
        });
    const auto gt_end = std::copy_if(std::execution::par_unseq, ground_truth.begin(), ground_truth.end(), class_gt.begin(),
        [class_id](const Detection& detection)
        {
            return detection.class_id == class_id;
        });
    class_pred.resize(static_cast<std::size_t>(pred_end - class_pred.begin()));
    class_gt.resize(static_cast<std::size_t>(gt_end - class_gt.begin()));

    if (class_gt.empty())
    {
        return 0.0F;
    }

    std::sort(std::execution::par_unseq, class_pred.begin(), class_pred.end(),
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
        precision[index] = dl::guarded_div(running_tp, running_tp + running_fp);
        recall[index] = dl::guarded_div(running_tp, gt_count);
    }

    std::vector<int> voc_points(kVocPoints);
    std::iota(voc_points.begin(), voc_points.end(), 0);
    const float average_precision = std::transform_reduce(std::execution::par_unseq, voc_points.begin(), voc_points.end(),
        0.0F, std::plus<float>(),
        [&](int point) -> float
        {
            const float recall_threshold = static_cast<float>(point) / static_cast<float>(kVocPoints - 1);
            if (class_pred.empty() && recall_threshold == 0.0F)
            {
                return 0.0F;
            }
            return std::transform_reduce(std::execution::par_unseq, precision.begin(), precision.end(), recall.begin(),
                0.0F,
                [](float lhs, float rhs)
                {
                    return std::max(lhs, rhs);
                },
                [recall_threshold](float precision_value, float recall_value) -> float
                {
                    return recall_value >= recall_threshold ? precision_value : 0.0F;
                });
        });
    return dl::guarded_div(average_precision, static_cast<float>(kVocPoints));
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
    return dl::guarded_div(intersection, union_area);
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

    std::vector<int> class_ids(classes.begin(), classes.end());
    const float counted = std::transform_reduce(std::execution::par_unseq, class_ids.begin(), class_ids.end(), 0.0F,
        std::plus<float>(),
        [&](int class_id) -> float
        {
            const bool has_gt = std::any_of(std::execution::par_unseq, ground_truth.begin(), ground_truth.end(),
                [class_id](const Detection& detection)
                {
                    return detection.class_id == class_id;
                });
            return has_gt ? 1.0F : 0.0F;
        });
    if (counted == 0.0F)
    {
        return 0.0F;
    }

    const float sum_ap = std::transform_reduce(std::execution::par_unseq, class_ids.begin(), class_ids.end(), 0.0F,
        std::plus<float>(),
        [&](int class_id) -> float
        {
            const bool has_gt = std::any_of(std::execution::par_unseq, ground_truth.begin(), ground_truth.end(),
                [class_id](const Detection& detection)
                {
                    return detection.class_id == class_id;
                });
            if (!has_gt)
            {
                return 0.0F;
            }
            return average_precision_for_class(predicted, ground_truth, class_id, iou_threshold);
        });
    return dl::guarded_div(sum_ap, counted);
}
