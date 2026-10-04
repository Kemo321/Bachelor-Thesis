# PackedImageLoader.cpp

Classification loader from a `DLIMG001` file. The whole split is read once into RAM as bytes. A batch on the GPU is NCHW images in [0, 1] and one-hot `[N, C]`. Prefetch converts `uint8` to float in the background so the GPU does not wait on that preparation. This path does not decode JPEG.

File layout, little-endian: 8 bytes of magic `DLIMG001`, then five `uint32` values (`n`, channels, height, width, class count), then `n * c * h * w` pixels as NCHW `uint8`, then `n` label bytes.

## read_u32_le

Assembles four bytes into a little-endian `uint32`. The file header is in that byte order.

## default_class_names

Builds the names `"0"` … `"C-1"`. The file carries the label number and does not carry class names.

## PackedImageLoader::PackedImageLoader

Requires a positive batch size. Opens the file, checks the magic, and reads the 20-byte header. A zero dimension or truncated data throws. Pixels and labels land in host vectors, class names come from `default_class_names`, then it logs and calls `reset`. The hot path therefore does not decode an image from disk on every batch.

## PackedImageLoader::~PackedImageLoader

Calls `join_prefetch` so the conversion thread finishes before the pixel buffer disappears.

## PackedImageLoader::reset

Finishes a pending prefetch, sets indices 0…n−1, and shuffles them when `shuffle` is set. The cursor returns to zero, and `launch_prefetch` starts conversion of the first batch.

## PackedImageLoader::has_next

Returns true when indices remain or the prefetch future is valid. The cursor is advanced when conversion starts, so the last batch is visible through the future.

## PackedImageLoader::size

Returns `n` from the file header.

## PackedImageLoader::batch_size

Returns the batch size from the constructor. The last batch of an epoch can be shorter. `take_indices` computes that length.

## PackedImageLoader::num_classes

Returns the class count from the header. That is the one-hot width.

## PackedImageLoader::channels

Returns the channel count from the header.

## PackedImageLoader::height

Returns the image height from the header.

## PackedImageLoader::width

Returns the image width from the header.

## PackedImageLoader::class_names

Returns the names `"0"` … `"C-1"` set when the file was opened.

## PackedImageLoader::label_at

Returns the label byte as `int`, without converting it to one-hot. An index past `labels_` throws.

## PackedImageLoader::copy_sample_float

Copies one sample to float NCHW and divides each byte by 255, the same way as the batch from `get_batch`. An index past the sample count throws.

## PackedImageLoader::take_indices

When the cursor is at the end, it returns an empty vector. Otherwise it takes `min(remaining, batch_size)` indices from `order_` and advances the cursor.

## PackedImageLoader::decode_indices

Zeros the image buffer and the targets `[N, C]`. For each sample, `dl::parallel_for` divides the pixels by 255 and places the one-hot 1. The label is clamped to `0…C-1`, so the write stays inside the row. An empty index list returns without work.

## PackedImageLoader::upload_host_batch

Uploads through `from_host` on the given stream, in `float32` and on the GPU. Images have shape `[N, channels, height, width]` from the header, and targets `[N, class count]`.

## PackedImageLoader::launch_prefetch

Takes the indices. An empty list starts nothing. Otherwise `std::async` with `std::launch::async` calls `decode_indices` immediately, so the `uint8`-to-float conversion does not wait for the GPU step to finish.

## PackedImageLoader::join_prefetch

When the future is not valid, it returns. `wait` blocks until the job ends, then the future is cleared. An exception from the job comes out of `get` in `get_batch`. It does not come out of `wait`. Dropping the future without `get` discards the result, so `reset` and the destructor do not propagate it.

## PackedImageLoader::get_batch

Marks the NVTX range `PackedImageLoader_GetBatch`. It collects the prefetch, and when there is none it throws because no samples remain. After `get` it clears the future and immediately starts the next conversion, still before `from_host`. Preparation of batch N+1 overlaps the upload of batch N and the GPU compute. A result with `n <= 0` throws a separate error.
