# bench_micro_more.cpp

Google Benchmark cases linked into the `bench_micro_ops` program (this file is linked together with `bench_micro_ops.cpp`). The `MICRO_BENCH` macro sets manual time and the millisecond unit. That iteration time is the `Profiler` GPU interval, not the Google Benchmark wall clock. The timer is `Profiler`: CUDA events, and `stop` synchronizes the GPU. LibTorch additionally calls `torch::cuda::synchronize()` inside the measured loop. After the loop, the float32 input bytes and the `VRAM_MiB` counter are recorded. Missing CUDA skips the case through `micro_require_cuda`.

The batch is `kMicroBatch` (16). Custom layers are warmed up with `kMicroWarmup` (5) forward calls.

On the custom backward path the measured interval starts at `backward`; that iteration's forward is earlier. LibTorch in the measured interval calls `backward` with the graph retained, zeroes the gradient, and synchronizes the device.

## repo_root

Returns the repository root. When `DEEPLEARN_SOURCE_DIR` is defined it uses that path; otherwise two levels above the working directory. CIFAR-10 and the CSV path are built from this root.

## bytes_of

Byte count of a `dl::Tensor` at `sizeof(float)`. That size is passed to the throughput counter as the bytes processed in one iteration.

## cifar_loader

Once per process, opens a `ClassificationLoader` on `data/cifar10/train` with batch `kMicroBatch`, side 32, and shuffling disabled. A missing `train` directory or an exception while opening leaves a null pointer.

## BM_Tensor_Add_Custom, BM_Tensor_Add_Torch

`BM_Tensor_Add_Custom` and `BM_Tensor_Add_Torch` measure the elementwise add of two tensors `[kMicroBatch, 512, 28, 28]`. The byte counter is the size of one operand. Custom calls `operator+` on `dl::Tensor`; LibTorch adds CUDA tensors.

## BM_Tensor_Mul_Custom, BM_Tensor_Mul_Torch

`BM_Tensor_Mul_Custom` and `BM_Tensor_Mul_Torch` measure the elementwise multiply of the same map `[kMicroBatch, 512, 28, 28]`. The shape is shared with the add so the cost of both elementwise ops can be compared on the same activation.

## BM_Tensor_Matmul_FCGrad_Custom, BM_Tensor_Matmul_FCGrad_Torch

`BM_Tensor_Matmul_FCGrad_Custom` and `BM_Tensor_Matmul_FCGrad_Torch` measure the matrix product `(7*7*1024, batch)` × `(batch, 4096)`. With batch 16 that is the head weight-gradient GEMM `50176 x 16 x 4096`. The left factor is the transposed input of the first fully connected head layer (`7*7*1024` → `4096`), and the right factor is the output gradient. The byte counter sums both operands.

## BM_Tensor_Transpose_Custom, BM_Tensor_Transpose_Torch

`BM_Tensor_Transpose_Custom` and `BM_Tensor_Transpose_Torch` measure the transpose of `[kMicroBatch, 7*7*1024]`. The result has the layout of the left factor in the `Matmul_FCGrad` pair. LibTorch materializes the transpose with `transpose(0, 1).contiguous()`.

## BM_Tensor_Sum_Custom, BM_Tensor_Sum_Torch

`BM_Tensor_Sum_Custom` and `BM_Tensor_Sum_Torch` measure the reduction of the whole map `[kMicroBatch, 512, 28, 28]` to a scalar. The shape is the same as for add and multiply.

## BM_Tensor_Clamp_Custom, BM_Tensor_Clamp_Torch

`BM_Tensor_Clamp_Custom` and `BM_Tensor_Clamp_Torch` measure clamping the map `[kMicroBatch, 1024, 7, 7]` to `[-1, 1]`. Clamping the `7×7×30` loss gradient is the separate `ClipGrad` pair.

## BM_MaxPool_Fwd_Custom, BM_MaxPool_Fwd_Torch

`BM_MaxPool_Fwd_Custom` and `BM_MaxPool_Fwd_Torch` measure `2×2` pooling with stride 2 on the input `[kMicroBatch, 192, 112, 112]`. That is the pooling after the 192-channel block.

## BM_MaxPool_Bwd_Custom, BM_MaxPool_Bwd_Torch

`BM_MaxPool_Bwd_Custom` and `BM_MaxPool_Bwd_Torch` measure the `2×2` pooling gradient for a `192×112×112` input.

## BM_LeakyReLU_Fwd_Custom, BM_LeakyReLU_Fwd_Torch

`BM_LeakyReLU_Fwd_Custom` and `BM_LeakyReLU_Fwd_Torch` measure LeakyReLU with slope `0.1` on the map `[kMicroBatch, 192, 112, 112]`.

## BM_LeakyReLU_Bwd_Custom, BM_LeakyReLU_Bwd_Torch

`BM_LeakyReLU_Bwd_Custom` and `BM_LeakyReLU_Bwd_Torch` measure the LeakyReLU `0.1` gradient for the same `192×112×112` map.

## BM_Flatten_Fwd_Custom, BM_Flatten_Fwd_Torch

`BM_Flatten_Fwd_Custom` and `BM_Flatten_Fwd_Torch` measure flattening `[kMicroBatch, 1024, 7, 7]` to a vector of length `7*7*1024`. That is the last backbone map before the first FC layer. Custom goes through the `Flatten` layer; LibTorch uses `view`.

## BM_Flatten_Bwd_Custom

`BM_Flatten_Bwd_Custom` measures the flatten gradient back to the shape `[kMicroBatch, 1024, 7, 7]`. This file has no LibTorch pair for flatten backward alone.

## BM_Softmax_Fwd_Custom, BM_Softmax_Fwd_Torch

`BM_Softmax_Fwd_Custom` and `BM_Softmax_Fwd_Torch` measure softmax along 10 classes, shape `[kMicroBatch, 10]`. The `SimpleCNN` head built in this file with 10 classes has that width. LibTorch calls `torch::softmax` on axis 1.

## BM_Softmax_Bwd_Custom, BM_Softmax_Bwd_Torch

`BM_Softmax_Bwd_Custom` and `BM_Softmax_Bwd_Torch` measure the softmax gradient for the vector `[kMicroBatch, 10]`.

## BM_CE_Fwd_Custom, BM_CE_Fwd_Torch

`BM_CE_Fwd_Custom` and `BM_CE_Fwd_Torch` measure the cross-entropy of logits `[kMicroBatch, 10]` for class 3. Custom receives a one-hot float tensor and calls `CrossEntropyLoss::loss`. LibTorch receives `long` indices and calls `cross_entropy`.

## BM_CE_Bwd_Custom

`BM_CE_Bwd_Custom` measures the cross-entropy derivative for the same class-3 one-hot (`CrossEntropyLoss::loss_derivative`). This file has no LibTorch pair for the derivative alone.

## BM_ClipGrad_Custom, BM_ClipGrad_Torch

`BM_ClipGrad_Custom` and `BM_ClipGrad_Torch` measure clamping a loss gradient of shape `[kMicroBatch, 7, 7, 30]` to a threshold of 10. The input is filled with the constant 5, so every value already lies in that range; the operation itself is what is measured. Custom calls `Network::clip_loss_gradient` with threshold `10`. LibTorch calls `torch::clamp` to `[-10, 10]`.

## BM_SimpleCNN_Fwd_Custom, BM_SimpleCNN_Fwd_Torch

`BM_SimpleCNN_Fwd_Custom` and `BM_SimpleCNN_Fwd_Torch` measure the forward of a 10-class network on the image `[kMicroBatch, 3, 32, 32]` (CIFAR side). Custom calls `forward_logits`, so Softmax is outside the measurement. LibTorch rebuilds the same stack in `Sequential`: two `3×3` convolution blocks (16, then 32 channels), LeakyReLU `0.1`, `2×2` pooling, `Flatten`, and a linear layer `32*8*8 → 10`.

## BM_Loader_CIFAR_Custom

`BM_Loader_CIFAR_Custom` measures `ClassificationLoader::get_batch` for CIFAR-10 (32×32 images, batch `kMicroBatch`). When the loader reaches the end, `reset` restarts the split. A missing `train` directory skips the case. The byte counter is the size of the image tensor.

## BM_Loader_CSV_Custom

`BM_Loader_CSV_Custom` measures construction of `CSVLoader` on `data/tabular/demo.csv`: one target column, header skipped, and a CUDA synchronization at the end. A missing file skips the case. The `VRAM_MiB` counter is recorded on every iteration. The `size()` call keeps the loader alive until the end of the iteration.
