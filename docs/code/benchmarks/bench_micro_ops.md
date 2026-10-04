# bench_micro_ops.cpp

Microbenchmarks of YOLOv1 image transfers, the `64→192` convolution, the first head FC layer, and the loss on the `7×7` grid. Together with `bench_micro_more.cpp` this file builds the `bench_micro_ops` program. Every case uses manual time and the millisecond unit. That iteration time is the `Profiler` GPU interval, not the Google Benchmark wall clock. `BENCHMARK_MAIN()` is at the end of the file.

Constants at batch 16: image `3×448`, convolution map `112×112`, first FC `7*7*1024 → 4096` (the head GEMM `50176 x 16 x 4096`), grid `7×7` with depth `10+20`. The convolution case is batch 16, `64→192`, kernel 3, spatial 112. Warmup is `kWarmup` (5). The timer is `Profiler`. Missing CUDA skips the case. LibTorch synchronizes the device inside the measured loop. On custom backward the measured interval starts at `backward` (that iteration's forward is earlier). LibTorch in that interval calls `backward` with the graph retained, zeroes the gradient, and synchronizes CUDA.

## require_cuda

Returns false and skips the case when `torch::cuda::is_available()` is false. The skip message says CUDA is required.

## numel

Product of the shape dimensions. Used for host buffer sizes and for the byte counter (`numel * sizeof(float)`).

## host_filled

Host vector of the requested shape, filled with one constant. Custom variants build GPU tensors from it.

## run_gpu_loop

Google Benchmark loop: `Profiler` around the body, iteration time from GPU milliseconds, a processed-byte counter, and `VRAM_MiB` after the loop.

## warmup_custom_conv

Calls `Conv2d::forward` `kWarmup` times and synchronizes the device. The warmup result is outside the measured interval.

## BM_H2D_Custom_FromHost, BM_H2D_Torch_To

`BM_H2D_Custom_FromHost` and `BM_H2D_Torch_To` measure the upload of the image `[16, 3, 448, 448]` from host to GPU, including device-buffer allocation. Custom calls `dl::Tensor::from_host` every time. LibTorch calls `to(CUDA)` on a CPU tensor filled with the constant `0.25`. One warmup upload runs before the loop. The byte counter is the image size.

## BM_H2D_Custom_ReuseMemcpy, BM_H2D_Torch_CopyInto

`BM_H2D_Custom_ReuseMemcpy` and `BM_H2D_Torch_CopyInto` measure the upload of the same `16×3×448` image into a GPU buffer that already exists. Custom copies with `cudaMemcpyAsync` from pinned memory on the current stream. LibTorch calls `copy_` into a tensor created by `empty_like`. Both sides do one copy before measurement so the buffer is initialized.

## BM_D2H_Custom_ToHost, BM_D2H_Torch_Cpu

`BM_D2H_Custom_ToHost` and `BM_D2H_Torch_Cpu` measure the read of the image `[16, 3, 448, 448]` from GPU to host. Custom calls `to_host`; LibTorch calls `cpu()`. One warmup read runs before the loop. The byte counter is the image size.

## BM_Conv2d_Fwd_Custom, BM_Conv2d_Fwd_Torch

`BM_Conv2d_Fwd_Custom` and `BM_Conv2d_Fwd_Torch` measure the forward of the convolution `64→192`, kernel `3×3`, padding 1, on the map `[16, 64, 112, 112]`. That is the shape of the second YOLOv1 block (64 channels after the first pooling): batch 16, 64 to 192, kernel 3, spatial 112. Custom warms up through `warmup_custom_conv`. LibTorch is in eval mode, with `setBenchmarkCuDNN` and `NoGradGuard`, and warms up with a `kWarmup` loop.

## BM_Conv2d_Bwd_Custom, BM_Conv2d_Bwd_Torch

`BM_Conv2d_Bwd_Custom` and `BM_Conv2d_Bwd_Torch` measure the gradient of the same `64→192` convolution. Custom computes forward on every iteration and synchronizes it before the timer; the measured interval is `backward` alone, with the output gradient filled with ones. LibTorch is in train mode: forward runs once before the loop, and the measured interval is `backward` with the graph retained, `zero_grad`, and synchronization. The byte counter is the size of the output gradient.

## BM_FC_Fwd_Custom, BM_FC_Fwd_Torch

`BM_FC_Fwd_Custom` and `BM_FC_Fwd_Torch` measure the forward of the layer `7*7*1024 → 4096` on the input `[16, 50176]`. That is the first linear layer of the YOLOv1 head. The three sizes are the head GEMM `50176 x 16 x 4096`. Both variants warm up `kWarmup` times. LibTorch is in eval mode, under `NoGradGuard`.

## BM_FC_Bwd_Custom, BM_FC_Bwd_Torch

`BM_FC_Bwd_Custom` and `BM_FC_Bwd_Torch` measure the gradient of the `50176→4096` layer. The weight-gradient product is the head GEMM `50176 x 16 x 4096`. Custom calls forward before the timer on each iteration and measures `backward` with the output gradient filled with ones. LibTorch is in train mode and, in the measured interval, calls `backward`, `zero_grad`, and synchronization. The byte counter is the size of the output gradient.

## BM_YOLOLoss_Fwd_Custom, BM_YOLOLoss_Fwd_Torch

`BM_YOLOLoss_Fwd_Custom` and `BM_YOLOLoss_Fwd_Torch` measure the YOLOv1 loss on the tensor `[16, 7, 7, 30]`. The prediction is filled with the constant `0.2`. In every batch sample one object sits in cell `(3, 3)`: box center and size, confidence 1, and class 0 (channel 10, the first class after the two boxes). Custom calls `YOLOLoss::loss`. LibTorch calls `compute_yolo_loss` under `NoGradGuard`; `make_torch_yolo_tensors` builds the tensors.

## BM_YOLOLoss_Bwd_Custom, BM_YOLOLoss_Bwd_Torch

`BM_YOLOLoss_Bwd_Custom` and `BM_YOLOLoss_Bwd_Torch` measure the derivative of the same loss with respect to the prediction `[16, 7, 7, 30]`, with the same object in cell `(3, 3)`. Custom calls `YOLOLoss::loss_derivative`. LibTorch calls `backward` on the loss scalar, with the graph retained, and zeroes the prediction gradient.

## make_torch_yolo_tensors

Builds a pair of LibTorch tensors on CUDA: a prediction of `0.2` with gradient enabled, and a zero target whose cell `(3, 3)` has the same box and class-0 description as the Custom variant. `BM_YOLOLoss_Fwd_Torch` and `BM_YOLOLoss_Bwd_Torch` use the pair.
