# bench_micro_more.cpp

Google Benchmark cases linked into the `bench_micro_ops` program (this file is linked together with `bench_micro_ops.cpp`). The `MICRO_BENCH` macro sets manual time and the millisecond unit. That iteration time is the `Profiler` GPU interval, not the Google Benchmark wall clock. The timer is `Profiler`: CUDA events, and `stop` synchronizes the GPU. LibTorch additionally calls `torch::cuda::synchronize()` inside the measured loop. After the loop, the float32 input bytes and the `VRAM_MiB` counter are recorded. Missing CUDA skips the case through `micro_require_cuda`. The exception is `BM_mAP_Custom`, which computes mAP on the CPU.

The batch is `kMicroBatch` (16), and the YOLOv1 image side is `kMicroImage` (448). Custom layers are warmed up with `kMicroWarmup` (5) forward calls. On the LibTorch side, BatchNorm, MaxPool, and FusedCBR have that loop. LeakyReLU, Dropout, Softmax, flatten, and the second head FC layer have one call before measurement. The full YOLO model warms up `kMicroModelWarmup` (2) times.

On the custom backward path the measured interval starts at `backward`; that iteration's forward is earlier. LibTorch in the measured interval calls `backward` with the graph retained, zeroes the gradient, and synchronizes the device.

## repo_root

Returns the repository root. When `DEEPLEARN_SOURCE_DIR` is defined it uses that path; otherwise two levels above the working directory. VOC, CIFAR-10, and the CSV path are built from this root.

## bytes_of

Byte count of a `dl::Tensor` at `sizeof(float)`. That size is passed to the throughput counter as the bytes processed in one iteration.

## voc_split

Once per process, splits `data/VOCdevkit/VOC2012` through `split_dataset` and keeps the training paths in memory. If the directory is missing or the image list is empty, `ready` stays false and the VOC loaders skip the case.

## cifar_loader

Once per process, opens a `ClassificationLoader` on `data/cifar10/train` with batch `kMicroBatch`, side 32, and shuffling disabled. A missing `train` directory or an exception while opening leaves a null pointer.

## voc_loader

Once per process, builds a `CustomDataLoader` on the VOC training split, with batch `kMicroBatch` and `is_train = false` (no augmentation and no shuffle on `reset`).

## BM_Tensor_Add_Custom, BM_Tensor_Add_Torch

`BM_Tensor_Add_Custom` and `BM_Tensor_Add_Torch` measure the elementwise add of two tensors `[kMicroBatch, 512, 28, 28]`. That is the 512-channel map after the third YOLOv1 pooling (side 28 for a 448 input). The byte counter is the size of one operand. Custom calls `operator+` on `dl::Tensor`; LibTorch adds CUDA tensors.

## BM_Tensor_Mul_Custom, BM_Tensor_Mul_Torch

`BM_Tensor_Mul_Custom` and `BM_Tensor_Mul_Torch` measure the elementwise multiply of the same map `[kMicroBatch, 512, 28, 28]`. The shape is shared with the add so the cost of both elementwise ops can be compared on the same activation.

## BM_Tensor_Matmul_FCGrad_Custom, BM_Tensor_Matmul_FCGrad_Torch

`BM_Tensor_Matmul_FCGrad_Custom` and `BM_Tensor_Matmul_FCGrad_Torch` measure the matrix product `(7*7*1024, batch)` × `(batch, 4096)`. With batch 16 that is the head weight-gradient GEMM `50176 x 16 x 4096`. The left factor is the transposed input of the first fully connected head layer (`7*7*1024` → `4096`), and the right factor is the output gradient. The byte counter sums both operands.

## BM_Tensor_Matmul_Head_Custom, BM_Tensor_Matmul_Head_Torch

`BM_Tensor_Matmul_Head_Custom` and `BM_Tensor_Matmul_Head_Torch` measure the product `(batch, 4096)` × `(4096, 1470)`. `1470 = 7*7*30`, and `30 = 10 + 20` is the YOLOv1 cell depth for 20 classes (two boxes and the classes). This is the GEMM of the second linear head layer.

## BM_Tensor_Transpose_Custom, BM_Tensor_Transpose_Torch

`BM_Tensor_Transpose_Custom` and `BM_Tensor_Transpose_Torch` measure the transpose of `[kMicroBatch, 7*7*1024]`. The result has the layout of the left factor in the `Matmul_FCGrad` pair. LibTorch materializes the transpose with `transpose(0, 1).contiguous()`.

## BM_Tensor_Sum_Custom, BM_Tensor_Sum_Torch

`BM_Tensor_Sum_Custom` and `BM_Tensor_Sum_Torch` measure the reduction of the whole map `[kMicroBatch, 512, 28, 28]` to a scalar. The shape is the same as for add and multiply.

## BM_Tensor_Clamp_Custom, BM_Tensor_Clamp_Torch

`BM_Tensor_Clamp_Custom` and `BM_Tensor_Clamp_Torch` measure clamping the map `[kMicroBatch, 1024, 7, 7]` to `[-1, 1]`. That is the last YOLOv1 convolutional map, just before flattening. Clamping the `7×7×30` loss gradient is the separate `ClipGrad` pair.

## BM_BN_Fwd_Custom, BM_BN_Fwd_Torch

`BM_BN_Fwd_Custom` and `BM_BN_Fwd_Torch` measure the `BatchNorm2d` forward on 192 channels and a `112×112` map, in train mode (batch statistics). The shape `[kMicroBatch, 192, 112, 112]` is the output of the `64→192` block before the second pooling.

## BM_BN_Bwd_Custom, BM_BN_Bwd_Torch

`BM_BN_Bwd_Custom` and `BM_BN_Bwd_Torch` measure the gradient of the same `BatchNorm2d` (192 channels, `112×112`, train mode). The timer range is described at the top of the file.

## BM_MaxPool_Fwd_Custom, BM_MaxPool_Fwd_Torch

`BM_MaxPool_Fwd_Custom` and `BM_MaxPool_Fwd_Torch` measure `2×2` pooling with stride 2 on the input `[kMicroBatch, 192, 112, 112]`. That is the pooling after the 192-channel block.

## BM_MaxPool_Bwd_Custom, BM_MaxPool_Bwd_Torch

`BM_MaxPool_Bwd_Custom` and `BM_MaxPool_Bwd_Torch` measure the `2×2` pooling gradient for a `192×112×112` input.

## BM_LeakyReLU_Fwd_Custom, BM_LeakyReLU_Fwd_Torch

`BM_LeakyReLU_Fwd_Custom` and `BM_LeakyReLU_Fwd_Torch` measure LeakyReLU with slope `0.1` on the map `[kMicroBatch, 192, 112, 112]`. That is the slope used in the YOLOv1 blocks.

## BM_LeakyReLU_Bwd_Custom, BM_LeakyReLU_Bwd_Torch

`BM_LeakyReLU_Bwd_Custom` and `BM_LeakyReLU_Bwd_Torch` measure the LeakyReLU `0.1` gradient for the same `192×112×112` map.

## BM_Dropout_Fwd_Custom, BM_Dropout_Fwd_Torch

`BM_Dropout_Fwd_Custom` and `BM_Dropout_Fwd_Torch` measure dropout `p = 0.5` in train mode on the vector `[kMicroBatch, 4096]`. Width 4096 is the YOLOv1 head hidden layer, after the first linear layer.

## BM_Dropout_Bwd_Custom, BM_Dropout_Bwd_Torch

`BM_Dropout_Bwd_Custom` and `BM_Dropout_Bwd_Torch` measure the gradient of that dropout (`p = 0.5`, train mode, vector of 4096).

## BM_Flatten_Fwd_Custom, BM_Flatten_Fwd_Torch

`BM_Flatten_Fwd_Custom` and `BM_Flatten_Fwd_Torch` measure flattening `[kMicroBatch, 1024, 7, 7]` to a vector of length `7*7*1024`. That is the last backbone map before the first FC layer. Custom goes through the `Flatten` layer; LibTorch uses `view`.

## BM_Flatten_Bwd_Custom

`BM_Flatten_Bwd_Custom` measures the flatten gradient back to the shape `[kMicroBatch, 1024, 7, 7]`. This file has no LibTorch pair for flatten backward alone.

## BM_Softmax_Fwd_Custom, BM_Softmax_Fwd_Torch

`BM_Softmax_Fwd_Custom` and `BM_Softmax_Fwd_Torch` measure softmax along 10 classes, shape `[kMicroBatch, 10]`. The `SimpleCNN` head built in this file with 10 classes has that width. LibTorch calls `torch::softmax` on axis 1.

## BM_Softmax_Bwd_Custom, BM_Softmax_Bwd_Torch

`BM_Softmax_Bwd_Custom` and `BM_Softmax_Bwd_Torch` measure the softmax gradient for the vector `[kMicroBatch, 10]`.

## BM_FusedCBR_Fwd_Custom, BM_FusedCBR_Fwd_Torch

`BM_FusedCBR_Fwd_Custom` and `BM_FusedCBR_Fwd_Torch` measure the `64→192` block, kernel `3×3`, padding 1, LeakyReLU `0.1`, on the input `[kMicroBatch, 64, 112, 112]` (the second YOLOv1 block, after the first pooling). Custom calls `FusedCBR2d`. LibTorch builds the same block from `Conv2d`, `BatchNorm2d`, and `LeakyReLU` and turns on `setBenchmarkCuDNN`. Both variants are in train mode.

## BM_FusedCBR_Bwd_Custom, BM_FusedCBR_Bwd_Torch

`BM_FusedCBR_Bwd_Custom` and `BM_FusedCBR_Bwd_Torch` measure the gradient of that `64→192` block on the `112×112` map. On the LibTorch side the gradient passes through the three separate layers of the sequence.

## BM_FusedCBR_Stem_Fwd_Custom, BM_FusedCBR_Stem_Fwd_Torch

`BM_FusedCBR_Stem_Fwd_Custom` and `BM_FusedCBR_Stem_Fwd_Torch` measure the first YOLOv1 layer: convolution `3→64`, kernel `7×7`, stride 2, padding 3, LeakyReLU `0.1`, on the image `[kMicroBatch, 3, kMicroImage, kMicroImage]`. LibTorch again composes convolution, batch-norm, and LeakyReLU as a sequence, with `setBenchmarkCuDNN`.

## BM_FC_Head2_Fwd_Custom, BM_FC_Head2_Fwd_Torch

`BM_FC_Head2_Fwd_Custom` and `BM_FC_Head2_Fwd_Torch` measure the linear layer `4096→1470` on the vector `[kMicroBatch, 4096]`. The output `1470 = 7*7*30` is the flattened detection grid for 20 classes. The LibTorch variant is in eval mode.

## BM_FC_Head2_Bwd_Custom, BM_FC_Head2_Bwd_Torch

`BM_FC_Head2_Bwd_Custom` and `BM_FC_Head2_Bwd_Torch` measure the gradient of the layer `4096→1470`. The LibTorch variant is in train mode.

## BM_MSE_Fwd_Custom, BM_MSE_Fwd_Torch

`BM_MSE_Fwd_Custom` and `BM_MSE_Fwd_Torch` measure the mean squared error of a prediction `[kMicroBatch, 10]` against a target of the same shape. Custom calls `MSELoss::loss`; LibTorch calls `torch::mse_loss`.

## BM_MSE_Bwd_Custom

`BM_MSE_Bwd_Custom` measures the MSE derivative with respect to the prediction `[kMicroBatch, 10]` (`MSELoss::loss_derivative`). This file has no LibTorch pair for the MSE derivative alone.

## BM_CE_Fwd_Custom, BM_CE_Fwd_Torch

`BM_CE_Fwd_Custom` and `BM_CE_Fwd_Torch` measure the cross-entropy of logits `[kMicroBatch, 10]` for class 3. Custom receives a one-hot float tensor and calls `CrossEntropyLoss::loss`. LibTorch receives `long` indices and calls `cross_entropy`.

## BM_CE_Bwd_Custom

`BM_CE_Bwd_Custom` measures the cross-entropy derivative for the same class-3 one-hot (`CrossEntropyLoss::loss_derivative`). This file has no LibTorch pair for the derivative alone.

## BM_ClipGrad_Custom, BM_ClipGrad_Torch

`BM_ClipGrad_Custom` and `BM_ClipGrad_Torch` measure clamping the YOLOv1 loss gradient of shape `[kMicroBatch, 7, 7, 30]` to a threshold of 10. The input is filled with the constant 5, so every value already lies in that range; the operation itself is what is measured. Custom calls `Network::clip_loss_gradient` with threshold `10`. LibTorch calls `torch::clamp` to `[-10, 10]`.

## BM_YOLO_Fwd_Custom, BM_YOLO_Fwd_Torch

`BM_YOLO_Fwd_Custom` and `BM_YOLO_Fwd_Torch` measure the forward of the full network for 20 classes in eval mode. The input is `[kMicroBatch, 3, kMicroImage, kMicroImage]`. Custom walks the layers of `YOLO::forward`; LibTorch uses `YOLOv1::forward` with `setBenchmarkCuDNN` and `NoGradGuard`. Warmup is `kMicroModelWarmup`.

## BM_YOLO_TrainStep_Custom, BM_YOLO_TrainStep_Torch

`BM_YOLO_TrainStep_Custom` and `BM_YOLO_TrainStep_Torch` measure one YOLOv1 training step (20 classes) on a synthetic target `[kMicroBatch, 7, 7, 30]` filled with zeros. The learning rate is `1e-4`. Layers are in train mode, so head dropout is active.

Custom computes `YOLOLoss`, clips the loss gradient at threshold 10, walks `get_all_layers` backward, and calls `step`. LibTorch uses `torch::optim::SGD`, `compute_yolo_loss`, `backward`, and `step`, and the measured interval also includes a CUDA synchronization. Warmup is `kMicroModelWarmup` full steps.

## BM_SimpleCNN_Fwd_Custom, BM_SimpleCNN_Fwd_Torch

`BM_SimpleCNN_Fwd_Custom` and `BM_SimpleCNN_Fwd_Torch` measure the forward of a 10-class network on the image `[kMicroBatch, 3, 32, 32]` (CIFAR side). Custom calls `forward_logits`, so Softmax is outside the measurement. LibTorch rebuilds the same stack in `Sequential`: two `3×3` convolution blocks (16, then 32 channels), LeakyReLU `0.1`, `2×2` pooling, `Flatten`, and a linear layer `32*8*8 → 10`.

## BM_Loader_VOC_Custom, BM_Loader_VOC_Torch

`BM_Loader_VOC_Custom` and `BM_Loader_VOC_Torch` measure the time to fetch one VOC2012 training batch, including image decode and the upload to the GPU. A missing VOC directory skips the case. Custom, when the split is exhausted, calls `reset` and takes `get_batch`. LibTorch builds a batch from `kMicroBatch` calls to `VOCYoloDataset::get` (the cursor wraps modulo the dataset size), stacks images and targets, and copies both stacks to CUDA. The byte counter uses the image elements. `VOCYoloDataset` is constructed with `is_train = false`.

## BM_Loader_CIFAR_Custom

`BM_Loader_CIFAR_Custom` measures `ClassificationLoader::get_batch` for CIFAR-10 (32×32 images, batch `kMicroBatch`). When the loader reaches the end, `reset` restarts the split. A missing `train` directory skips the case. The byte counter is the size of the image tensor.

## BM_Loader_CSV_Custom

`BM_Loader_CSV_Custom` measures construction of `CSVLoader` on `data/tabular/demo.csv`: one target column, header skipped, and a CUDA synchronization at the end. A missing file skips the case. The `VRAM_MiB` counter is recorded on every iteration. The `size()` call keeps the loader alive until the end of the iteration.

## BM_DecodeNMS_Custom

`BM_DecodeNMS_Custom` measures `decode_yolo_tensor` and `apply_nms` on a host vector of length `7*7*30`, filled with the constant `0.2`. The confidence threshold is `0.05`, the image side is 448, there are 20 classes, and the NMS IoU threshold is `0.5`. The case requires CUDA through `micro_require_cuda`; decode and NMS themselves run on host data. The result is passed to `DoNotOptimize`.

## BM_mAP_Custom

`BM_mAP_Custom` measures `mean_average_precision` on the CPU. The set is 32 ground truths and 64 predictions (two per index), IoU threshold `0.5`. The case does not check CUDA and does not record a VRAM counter. The mAP result is passed to `DoNotOptimize`.
