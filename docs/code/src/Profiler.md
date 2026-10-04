# Profiler.cpp

Times an interval on the GPU with a pair of CUDA events and reports VRAM use.

## Profiler::Profiler

When CUDA is enabled, creates `start_event_` and `stop_event_`. Without CUDA the constructor allocates nothing.

## Profiler::~Profiler

Destroys events that are not null and clears the pointers. The result of `cudaEventDestroy` is discarded, because the object is already shutting down.

## Profiler::start

Without CUDA, throws. Otherwise records the start event and sets `running_`.

## Profiler::stop

Without CUDA, or without a preceding `start()`, throws. Records the stop, synchronizes the host with `stop_event_` (the time between the events is valid only then), computes the milliseconds, clears `running_`, and returns the result.

## Profiler::get_vram_usage_mb

Without CUDA, throws. Reads free and total bytes from `cudaMemGetInfo`. When total is less than free, returns 0 so the unsigned subtraction does not wrap. Otherwise returns `(total - free) / 1024^2`.
