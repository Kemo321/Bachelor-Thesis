# bench_voc_custom.cpp

One epoch of manual YOLOv1 training on VOC2012, registered as a Google Benchmark case with real time. The batch arguments are 8 and 16. The LibTorch counterpart is `BM_YOLOv1_SingleEpochTraining` in `bench_voc_torch.cpp`.

## BM_CustomYOLO_ManualTraining

`BM_CustomYOLO_ManualTraining` measures a pass over the whole VOC2012 training split. Data comes from `../../data/VOCdevkit/VOC2012`. An empty image list skips the case. `CustomDataLoader` receives the batch from `state.range(0)` and `is_train = false`, so this epoch has no augmentation and no shuffle on `reset`.

The model is `YOLO` with the default class count (20). Layers go to the GPU. The `Network` constructor sets their `learning_rate` to `1e-4` and leaves the default gradient-clip threshold `0`, so `clip_loss_gradient` and `clip_parameter_gradients` do not change values.

In each batch: `forward`, a host download of the `YOLOLoss::loss` scalar (`to_host`, result discarded), the loss derivative, backward from the end of `get_all_layers`, then `step`. The timed loop includes `CustomDataLoader::get_batch` (decode and upload), not only the train step. `Profiler` covers the whole epoch, including the loader and the loss transfer.

`SetItemsProcessed` sums images from every benchmark iteration. `Img/Sec` is a rate counter against real time. `VRAM_MiB` is occupancy after the run. `GPU_ms` is the CUDA time of the last iteration, overwritten in the loop.
