# dataset.cpp

Preparation of the detection set and `CustomDataLoader`. The constant `VOC_CLASSES_DEFAULT` is the 20 Pascal VOC class names. A different `class_names` list changes the one-hot width in the grid. The default split fractions live in the `split_dataset` declaration: `train_ratio = 0.7`, `val_ratio = 0.15`.

The grid has 7×7 cells, two slots per cell, and five numbers per slot (`x`, `y`, `w`, `h`, objectness). The class one-hot starts at index 10.

## convert_voc_to_yolo

Reads XML from the annotation directory and writes `.txt` files beside the JPEGs in YOLO format: `class center_x center_y width height`, with all four numbers divided by the dimensions from the `size` node. The class name is stored as an index from the supplied list.

Skips files other than `.xml`, XML without a sibling `.jpg`, a failed read, and a width or height of zero. An empty `bndbox` and an unknown class name do not enter the file. The `.txt` file is created even when no box was written. When the count is greater than zero, it logs how many files were created.

## hwc_to_chw

Rewrites a `cv::Mat` in height–width–channel order into three channel–row–column planes. The network tensor and `from_host` expect NCHW, while OpenCV keeps the channels inside the pixel.

## apply_affine

Builds a 2×3 matrix with scale `scale` and a translation computed from `dx` and `dy`, then calls `warpAffine` with linear interpolation and a black border. Both translations use half the image width (`cols / 2`), not the height. On the training path the image is already a 448×448 square.

`load_sample` then passes the same `scale`, `dx`, and `dy` to `encode_targets`, so the box stays on the object after the image transform.

## apply_hsv_jitter

Converts through HSV, multiplies saturation and value by the given factors, and clamps both channels to [0, 1]. The only caller is `load_sample` when `is_train` is set, so evaluation does not change colors.

## encode_targets

Zeros a 7×7×(10 + class count) grid and reads successive quintuples `class cx cy w h` from the YOLO file. A class id outside the class range is skipped. The lambda `at` addresses one number in a cell.

When `is_train` is true, the box center and size pass through the inverse of the same scale and shift as `apply_affine`. A center outside [0, 1] does not enter the grid. Width and height are clamped to [0, 1]. The cell comes from the center times 7, clamped to 0…6.

The first slot of that cell whose objectness is still 0 receives `x` and `y` as the position inside the cell (`center * 7 - index`), `w` and `h` relative to the image, and objectness 1. The class one-hot lands at index 10 + id. When both slots of the cell are occupied, the box is skipped.

## split_dataset

Looks for `JPEGImages` and `Annotations` under `voc_root`. A missing directory ends in a log and a return, without an exception. It creates `labels`. If any `.jpg` lacks a `.txt`, it calls `convert_voc_to_yolo`.

It collects JPEG–label pairs, shuffles them with `mt19937` seeded from `random_device`, and cuts by index. The train end is `floor(n * train_ratio)`. The validation end is that index plus `floor(n * val_ratio)`. The remaining indices go to test. With the defaults 0.7 and 0.15, the third part is that remainder after truncation to a whole number of images. An empty pair list ends in a log and a return.

## CustomDataLoader::CustomDataLoader

Stores the paths, batch size, training flag, class count, and side length 448. Throws `std::runtime_error` when the batch size is not positive, the class list is empty, or the image and label counts differ. At the end it calls `reset`, which sets the epoch order and the first prefetch.

## CustomDataLoader::~CustomDataLoader

Calls `join_prefetch` so the decode thread finishes before the paths and the generator disappear.

## CustomDataLoader::reset

First finishes a pending prefetch so it does not read the old order. The cursor returns to zero, `order_` receives indices 0…n−1, and when `is_train` is set and the list is non-empty those indices are shuffled. Then `launch_prefetch` decodes the first batch before the GPU asks for data.

## CustomDataLoader::has_next

Returns true when the prefetch future is valid or the cursor has not reached the end of `order_`. The cursor alone is not enough: `take_job` advances it when prefetch starts, so the last batch is in flight and the cursor is already past it.

## CustomDataLoader::size

Returns the number of images in `paths_`.

## CustomDataLoader::batch_size

Returns the batch size from the constructor. The last batch of an epoch can be shorter. `take_job` computes that length.

## CustomDataLoader::load_sample

`imread` reads the JPEG. Failure leaves a zero CHW buffer and calls `encode_targets` with no scale or shift, so the batch row keeps a fixed shape. Otherwise the image goes from BGR to RGB, is resized to 448, and is divided by 255.

Augmentation runs only when `is_train` is set: scale from [0.8, 1.2], shifts `dx` and `dy` from [−0.2, 0.2], saturation and value from [0.66, 1.5]. Then `apply_affine` and `apply_hsv_jitter`. A non-continuous matrix is cloned, `hwc_to_chw` lays out the channels, and `encode_targets` receives the same scale and shift as the image.

## CustomDataLoader::take_job

When the cursor is at the end, it returns empty vectors. Otherwise it takes `min(remaining, batch_size)` indices from `order_`, advances the cursor, and draws a seed `rng_()` for each sample. A private seed per sample means the `decode_job` threads do not share one generator.

## CustomDataLoader::decode_job

Builds a `HostBatch`: CHW images and a grid of width 10 + class count. An empty index list returns immediately. `dl::parallel_for` calls `load_sample` with a local `mt19937` from the seed. The copy into the batch buffer happens only when the sample length matches the expected length. Otherwise that row stays the zeros from `assign`.

## CustomDataLoader::upload_host_batch

Builds a `Batch` with two `from_host` calls on the given stream, in `float32` and on the GPU. Images have shape `[N, 3, 448, 448]`, targets `[N, 7, 7, attributes]`.

## CustomDataLoader::launch_prefetch

Takes a job from `take_job`. Empty indices start nothing. Otherwise `std::async` with `std::launch::async` runs `decode_job` immediately, at this call, so the work is already running before the next `get`.

## CustomDataLoader::join_prefetch

When the future is not valid, it returns. Otherwise `wait` blocks until the job ends, and then the future is cleared. `wait` does not pull the exception stored in the future. That exception comes out of `get` in `get_batch`. Dropping the future without `get` discards that result, so `reset` and the destructor do not propagate it.

## CustomDataLoader::get_batch

Marks the NVTX range `DataLoader_GetBatch`. When prefetch is not running, it starts one. When there is still no future, it throws because no samples remain. `get` returns the host buffer, and the decode exception if one was stored. The future is cleared, and `launch_prefetch` starts the next batch before `from_host`. JPEG decode of batch N+1 therefore runs together with the upload of batch N and with the later GPU compute. An empty result (`n <= 0`) throws a separate error.
