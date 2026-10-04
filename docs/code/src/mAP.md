# mAP.cpp

Computes detection IoU and mean average precision (mAP) with the VOC 11-point method.

## box_area

Box area: `max(0, width) * max(0, height)`. A negative side yields zero, so the IoU cannot come out negative.

## average_precision_for_class

Keeps predictions and ground truth with the given `class_id`. No ground truth for that class returns 0. Sorts predictions by descending score. Matches each prediction greedily to a still-free ground-truth box of the highest IoU. IoU not less than the threshold is a true positive and that ground-truth box is taken; otherwise it is a false positive. Precision and recall are cumulative: `guarded_div(tp, tp + fp)` and `guarded_div(tp, ground-truth count)`.

AP is the mean of 11 recall thresholds `0, 0.1, …, 1`. At a threshold it takes the maximum precision among predictions whose recall has already reached the threshold. Empty predictions at threshold 0 return 0 for that point. The result divides the sum by 11. The IoU threshold is the one the caller passed in.

## detection_iou

IoU of two `Detection` values in `(x, y, width, height)` layout. The intersection is the product of the positive overlap width and height. The union comes from `box_area`, minus the intersection. `guarded_div` divides by `max(denominator, eps)`.

## mean_average_precision

A threshold outside `[0, 1]` throws. Empty ground truth returns 0. Collects class ids from the ground truth and from the predictions. The denominator is the number of classes that occur in the ground truth; when that is 0, returns 0. The numerator is the sum of `average_precision_for_class` for those classes, and the result is `guarded_div` of that sum by the denominator. Classes present only in the predictions are not included in the mean.
