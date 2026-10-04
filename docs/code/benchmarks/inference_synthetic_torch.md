# inference_synthetic_torch.cpp

YOLOv1 inference on the synthetic-shapes dataset in LibTorch. The custom-stack counterpart is `inference_synthetic_custom.cpp` (`main`).

## torch_to_host

Copies the tensor to the CPU, forces a contiguous float32 layout, then `memcpy` into a `std::vector<float>`. `decode_yolo_tensor` expects that vector.

## main

`main` reads the `synthetic_torch` configuration. The defaults are 3 classes, confidence threshold `0.10`, and NMS threshold `0.45`. The data directory is `data/Synthetic3/train`, and the results directory is `results/synthetic`. Class names: square, circle, triangle.

Weight load: the file `yolov1_synthetic_torch_final.pt` must exist. `torch::load` loads it into `YOLOv1`. The device is chosen by `torch::cuda::is_available()` (CUDA or CPU). The model is in eval mode.

Images come from the test split, and when that split is empty, from the training split. The order is shuffled, and at most 30 paths remain.

For each image, `prepare_yolo_input` produces NCHW 448×448 in RGB and the range `[0, 1]`. `from_blob` clones the buffer and sends it to the chosen device. `forward` runs under `NoGradGuard`, and the result passes through `torch_to_host`. The confidence threshold acts in `decode_yolo_tensor` (coordinates in the original image scale). NMS removes later boxes of the same class whose IoU is above the threshold. The program draws those kept boxes and does not compute mAP. The save writes green BGR boxes `(0, 255, 0)` to `predictions_torch/torch_<file>`.
