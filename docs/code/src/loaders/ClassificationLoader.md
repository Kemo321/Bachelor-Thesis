# ClassificationLoader.cpp

Classification loader from directories `root/<split>/<class>/*`. The image enters the GPU as NCHW in [0, 1], and the target as one-hot `[N, C]`. The order of the class names is the index of that 1. Prefetch decodes the next JPEG batch while the GPU computes the current one.

## is_image_file

Lowercases the extension and accepts `.jpg`, `.jpeg`, `.png`, and `.bmp`. Other files in a class folder do not enter the set.

## hwc_to_chw

Splits a `CV_32FC3` row into three planes. The pointer steps by three floats, because OpenCV keeps the channels interleaved in the pixel and the network reads NCHW.

## ClassificationLoader::ClassificationLoader

Requires a positive batch size and a positive image side. The split path is `root/split`. When that is not a directory, the loader reads directly from `root`. When `root` is not a directory either, it throws.

An empty `class_names` list is filled with subdirectory names, sorted. A list passed in from outside stays in the given order, including when a folder is missing, so the test one-hot has the same `C` as training. From each existing folder it collects path–class-index pairs. No classes, or no images, throws. At the end it logs and calls `reset`.

## ClassificationLoader::~ClassificationLoader

Calls `join_prefetch` so the decode thread finishes before the sample list disappears.

## ClassificationLoader::reset

Finishes a pending prefetch, zeros the cursor, and sets indices 0…n−1. When `shuffle` is set and the list is non-empty, it shuffles them. Then it starts prefetch of the first batch.

## ClassificationLoader::has_next

Returns true when the prefetch future is valid or indices remain. The cursor already moves in `take_indices`, so the last batch is visible through the future, with the cursor already past it.

## ClassificationLoader::size

Returns the number of collected images.

## ClassificationLoader::batch_size

Returns the batch size from the constructor. The last batch of an epoch can be shorter. `take_indices` computes that length.

## ClassificationLoader::num_classes

Returns the length of `class_names_`, which is the width of the one-hot vector.

## ClassificationLoader::image_size

Returns the side of the square that images are reduced to before `from_host`.

## ClassificationLoader::class_names

Returns the names in one-hot index order. The same list, passed to the test loader, keeps `C` fixed.

## ClassificationLoader::load_sample

Zeros the CHW buffer and reads the file as color BGR. An empty read returns the class id from the directory, and the image stays zeros. Otherwise the image goes to RGB, and when a side differs from `image_size` it is scaled linearly. Then it divides by 255, clones the matrix when it is not continuous, and calls `hwc_to_chw`. The return value is the class id.

## ClassificationLoader::take_indices

When the cursor is at the end, it returns an empty vector. Otherwise it takes `min(remaining, batch_size)` indices from `order_` and advances the cursor.

## ClassificationLoader::decode_indices

Builds an image buffer and a target buffer zeroed to `[N, C]`. `dl::parallel_for` calls `load_sample`. The image is copied only when it has the expected length. One 1 is placed in the target row at the class index clamped to `0…C-1`, so the write stays inside the one-hot.

## ClassificationLoader::upload_host_batch

Uploads through `from_host` on the given stream, in `float32` and on the GPU. Images have shape `[N, 3, side, side]`, targets `[N, C]`.

## ClassificationLoader::launch_prefetch

Takes the indices. An empty list starts nothing. Otherwise `std::async` with `std::launch::async` decodes the batch in the background so the GPU does not wait on JPEG.

## ClassificationLoader::join_prefetch

When the future is not valid, it returns. `wait` blocks until the job ends, then the future is cleared. A decode exception comes out of `get` in `get_batch`. It does not come out of `wait`. Dropping the future without `get` discards the result, so `reset` and the destructor do not propagate it.

## ClassificationLoader::get_batch

Marks the NVTX range `ClassificationLoader_GetBatch`. It collects the prefetch, and when there is none it throws because no samples remain. After `get` it clears the future and immediately starts the next batch, still before `from_host`. JPEG decode of batch N+1 overlaps the upload of batch N and the GPU compute. A result with `n <= 0` throws a separate error.
