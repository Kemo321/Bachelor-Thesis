# utils.cpp

Computes IoU of OpenCV boxes, applies NMS, unpacks a YOLO output into detections, and draws them on an image.

## calculate_iou

Intersection `box_a & box_b`, then the intersection area divided by the sum of the areas minus the intersection. `guarded_div` uses `max(denominator, eps)`, so the division is not by zero.

## apply_nms

Sorts `detections` by descending `score` (parallel execution). For same-class pairs in the triangle `i < j`, writes the IoU into a `count * count` matrix. Then walks from the highest score and marks as suppressed the later boxes whose IoU exceeds `nms_threshold`. Suppression is sequential because it depends on score order. Returns the detections that were kept. The input vector is left sorted.

## decode_yolo_tensor

Expects a flat buffer of at least `49 * (10 + num_classes)`, laid out as `[1, 7, 7, 10 + classes]`. A shorter buffer throws. The grid is 7×7, two boxes per cell, five numbers per box (`tx, ty, tw, th`, objectness), and class probabilities from offset 10.

The lambda `at` reads the element `(grid_i * 7 * attributes) + (grid_j * attributes) + offset`.

Each cell is computed in parallel. The class is the argmax of the probabilities (the search starts at `-1e6`). A box whose objectness is not greater than `conf_threshold` is dropped. The center in pixels is `(tx + column) / 7 * img_width` and `(ty + row) / 7 * img_height`. Width and height are `tw` and `th` multiplied by the image dimensions. The top-left corner is clamped to zero. Only slots marked valid reach the result, and that compacting happens after the parallel loop.

## draw_detections

Draws a rectangle of thickness 2 for each detection. When `class_names` has three entries and the first is `"square"`, class 0 is white, class 1 is green, and the rest is blue (BGR channels). Otherwise the color stays `default_color`. The label is the class name and the score cut to four characters, on a background of the box color, with black text.
