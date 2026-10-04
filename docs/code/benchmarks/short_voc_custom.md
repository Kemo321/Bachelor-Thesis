# short_voc_custom.cpp

A short YOLO detector run on Pascal VOC on the custom stack. The loop has 3 epochs stored in code as `kEpochs`. Thresholds, learning rate, directory, and VOC subset come from the `voc_custom` pipeline JSON. Images enter box scoring at resolution 448. Results go to `results/voc_short`.

## main

1. Reads the configuration of the full VOC experiment, but leaves the loop length at the three epochs from code.
2. Splits the directory with `split_dataset` (default 0.7 train, 0.15 val, the rest test) and the default class list from the header. Both loaders are created with training mode off, so this path does not add image augmentation.
3. Builds YOLO and the SGD trainer, moves the layers to the GPU, and sets the hyperparameters.
4. Opens the detection metrics CSV.
5. In each of the three epochs it sets the learning rate from the schedule, computes the YOLO loss, and updates the weights on the train list.
6. In eval mode it computes the loss on the test set and collects boxes.
7. Computes mAP at IoU 0.5 and appends a CSV row. After three epochs the program ends.
