# short_voc_torch.cpp

A short YOLOv1 detector run on Pascal VOC in LibTorch. The loop has 3 epochs stored in code as `kEpochs`. Thresholds, learning rate, worker count, directory, and VOC subset come from the `voc_torch` pipeline JSON. Images enter box scoring at resolution 448. Results go to `results/voc_short`. The device is CUDA or CPU.

## main

1. Reads the configuration of the full VOC experiment, but leaves the loop length at the three epochs from code.
2. Splits the directory with `split_dataset` (default 0.7 train, 0.15 val, the rest test). Both loaders are created with training mode off.
3. Builds YOLOv1, moves it to the selected device, and creates SGD.
4. Opens the detection metrics CSV.
5. In each of the three epochs it sets the learning rate from the schedule, computes the YOLO loss, and updates the weights on the train list.
6. In eval mode, with no gradient, it computes the loss on the test set and collects boxes, then returns to train mode.
7. Computes mAP at IoU 0.5 and appends a CSV row. After three epochs the program ends.
