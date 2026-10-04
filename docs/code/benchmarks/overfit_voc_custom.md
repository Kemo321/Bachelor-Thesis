# overfit_voc_custom.cpp

Overfits a YOLO detector on a few Pascal VOC images on the custom stack. The list of 20 classes is in this file. Epoch count, batch size, thresholds, and learning rate come from the `overfit_voc_custom` pipeline JSON. When the `epochs` key is missing, the code falls back to 300. Images enter the model at resolution 448.

## main

1. Reads the configuration. The data directory is the VOCdevkit root plus the subset from JSON (fallback `VOC2012`).
2. Splits the directory with `split_dataset`, then keeps at most one batch of the first images from the train list. An empty list ends the program. The loader for that batch has training mode off.
3. Builds YOLO and the SGD trainer, moves the layers to the GPU, and sets the hyperparameters.
4. Opens the detection metrics CSV.
5. Each epoch sets the learning rate from the schedule, computes the YOLO loss, and updates the weights on that same batch.
6. A second pass, already in eval mode, computes the loss and boxes on the same images.
7. Computes mAP at IoU 0.5 and appends a CSV row.
8. After the last epoch it saves the weights, then draws detections on the images from the batch and saves them next to the metrics.
