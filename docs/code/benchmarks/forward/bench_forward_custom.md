# bench_forward_custom.cpp

Times `forward_logits` for every case in `forward_latency`. The input is a constant `0.5` tensor allocated before the clock starts. Warmup runs are not timed. `cudaDeviceSynchronize` brackets the timed repeats.

## main

Writes `results/forward/metrics_custom.csv` with `Model;Batch;Repeats;Time(s);SamplesPerSec;VRAM_MiB`. `bench_forward_torch` writes the same columns for the LibTorch twins.
