# inference_synthetic_custom.cpp

YOLOv1 inference on the synthetic-shapes dataset with the custom stack. The LibTorch counterpart is `inference_synthetic_torch.cpp` (`main` and `torch_to_host`).

## main

`main` reads the `synthetic_custom` configuration. The defaults are 3 classes, confidence threshold `0.10`, and NMS threshold `0.45`. The data directory is `data/Synthetic3/train`, and the results directory is `results/synthetic`. Class names in grid-index order: square, circle, triangle.

Weight load: the file `yolov1_synthetic_custom_final.pt` must exist. `Network::load` writes it into the `YOLO` layers, then every layer moves to the GPU and into eval mode.

Images come from the test split, and when that split is empty, from the training split. The order is shuffled, and at most 30 paths remain. An empty set exits with code 1.

For each image, `prepare_yolo_input` scales the side to 448, converts BGR to RGB, divides by 255, and lays out NCHW. The tensor `1×3×448×448` goes to the GPU, and `forward` returns to the host. The confidence threshold acts in `decode_yolo_tensor` (coordinates in the original image scale). NMS removes later boxes of the same class whose IoU is above the threshold. The program draws those kept boxes and does not compute mAP. The save writes red BGR boxes `(0, 0, 255)` to `predictions_custom/custom_<file>`.
