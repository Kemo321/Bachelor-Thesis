# train_bccd_torch.cpp

Trains a YOLOv1 detector on the BCCD blood-cell set (RBC, WBC, and platelets) in LibTorch. Epoch count, loader worker count, detection thresholds, and learning rate come from the `bccd_torch` pipeline JSON. When the `epochs` key is missing, the code falls back to 800. Images enter the model at resolution 448.

## main

1. Reads the configuration. Selects CUDA or CPU, and on CUDA enables the cuDNN benchmark.
2. Splits the directory with `split_dataset` (default 0.7 train, 0.15 val, the rest test). The training loader shuffles the train list, and mAP walks the test list.
3. Builds YOLOv1, moves it to the selected device, and creates SGD.
4. Opens the detection metrics CSV.
5. Each epoch sets the learning rate from the schedule and, on the training set, computes the YOLO loss, runs backward, and takes an optimizer step.
6. In eval mode, with no gradient, it computes the loss on the test set and turns predictions and labels into boxes.
7. Computes mAP at IoU 0.5 and appends a CSV row.
8. After the last epoch it saves the LibTorch model weights.
