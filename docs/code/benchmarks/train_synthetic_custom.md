# train_synthetic_custom.cpp

Trains a YOLO detector on synthetic shapes (square, circle, triangle) on the custom stack. Epoch count, detection thresholds, and learning rate come from the `synthetic_custom` pipeline JSON. When the `epochs` key is missing, the code falls back to 800. Images enter the model at resolution 448.

## main

1. Reads the configuration and the image and results directories.
2. Splits the directory with `split_dataset` (default 0.7 train, 0.15 val, the rest test). Training walks the train list, and mAP walks the test list.
3. Builds YOLO and the SGD trainer, then moves the layers to the GPU.
4. Opens the detection metrics CSV.
5. Each epoch sets the learning rate from the schedule, computes the YOLO loss, and updates the weights on the training set.
6. In eval mode it computes the loss on the test set and turns predictions and labels into boxes.
7. Computes mAP at IoU 0.5 and appends a CSV row.
8. After the last epoch it saves the trainer weights.
