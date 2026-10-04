# inference_voc_custom.cpp

YOLOv1 inference on VOC images with the custom stack. The LibTorch counterpart is `inference_voc_torch.cpp` (`main` and `torch_to_host`).

## main

`main` reads the `voc_custom` configuration. The defaults are 20 classes, confidence threshold `0.25`, NMS threshold `0.5`, and subset `VOC2012`. With no arguments the weights are `results/voc/yolov1_voc_custom_final.pt`, and the images are `JPEGImages` of that subset. Three program arguments replace the weight path and the image file or directory. Any other argument count exits with code 1.

Weight load: the file must exist. `Network::load` writes it into the `YOLO` layers, then every layer moves to the GPU and into eval mode.

`collect_image_paths` for a directory sorts image files and keeps at most 50. A single file enters as one path.

For each image, `prepare_yolo_input` scales the side to 448, converts BGR to RGB, divides by 255, and lays out NCHW. `Tensor::from_host` builds the input `1×3×448×448` (default device GPU). The confidence threshold acts in `decode_yolo_tensor` (coordinates in the original image scale). NMS removes later boxes of the same class whose IoU is above the threshold. The program draws those kept boxes and does not compute mAP. The save writes red BGR boxes `(0, 0, 255)` to `predictions_custom/inference_<file>`.
