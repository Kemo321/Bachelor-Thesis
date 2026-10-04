# inference_voc_torch.cpp

YOLOv1 inference on VOC images in LibTorch. The custom-stack counterpart is `inference_voc_custom.cpp` (`main`).

## torch_to_host

Copies the tensor to the CPU, forces a contiguous float32 layout, then `memcpy` into a `std::vector<float>`. `decode_yolo_tensor` expects that vector.

## main

`main` reads the `voc_torch` configuration. The defaults are 20 classes, confidence threshold `0.25`, NMS threshold `0.5`, and subset `VOC2012`. With no arguments the weights are `results/voc/yolov1_voc_torch_final.pt`, and the images are `JPEGImages` of that subset. Three program arguments replace the weight path and the image file or directory. Any other argument count exits with code 1.

Weight load: the file must exist. `torch::load` loads it into `YOLOv1`. The device is chosen by `torch::cuda::is_available()` (CUDA or CPU). The model is in eval mode.

`collect_image_paths` for a directory sorts image files and keeps at most 50.

For each image, `prepare_yolo_input` produces NCHW 448×448 in RGB and the range `[0, 1]`. `from_blob` clones the buffer and sends it to the chosen device. `forward` runs under `NoGradGuard`, and the result passes through `torch_to_host`. The confidence threshold acts in `decode_yolo_tensor` (coordinates in the original image scale). NMS removes later boxes of the same class whose IoU is above the threshold. The program draws those kept boxes and does not compute mAP. The save writes green BGR boxes `(0, 255, 0)` to `predictions_torch/inference_<file>`.
