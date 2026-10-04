# bench_voc_torch.cpp

One YOLOv1 training epoch in LibTorch on VOC2012, registered as a Google Benchmark case with real time. The batch arguments are 8 and 16. The custom-stack counterpart is `BM_CustomYOLO_ManualTraining` in `bench_voc_custom.cpp`.

## BM_YOLOv1_SingleEpochTraining

`BM_YOLOv1_SingleEpochTraining` measures a pass over the whole VOC2012 training split. Data comes from `../../data/VOCdevkit/VOC2012`. An empty image list skips the case. The LibTorch loader uses the batch from `state.range(0)`, four workers, and `VOCYoloDataset` with the second argument `false` (no augmentation). The device is CUDA when `torch::cuda::is_available()` is true, otherwise CPU.

The model is `YOLOv1` with the default class count (20). The optimizer is Adam with learning rate `1e-4`. In each batch: `zero_grad`, `forward`, `compute_yolo_loss`, `backward`, `step`, and on CUDA a synchronization as well. The epoch loss sum is passed to `DoNotOptimize`.

`SetItemsProcessed` sums images from every benchmark iteration. `Img/Sec` is a rate counter against real time. After the loop, already outside the measured interval, the weights go to `../../results/yolov1_bench_epoch.pt`.
