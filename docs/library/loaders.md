# Loaders

A `Batch` is two GPU tensors: `images` and `targets`. Loaders own host decode and the upload. They do not run `forward` or `backward`.

| Class | Header | Source | `images` | `targets` |
| --- | --- | --- | --- | --- |
| `ClassificationLoader` | `loaders/ClassificationLoader.hpp` | `root/<split>/<class>/*.jpg` | NCHW, side length from the constructor | one-hot `[N, C]` |
| `PackedImageLoader` | `loaders/PackedImageLoader.hpp` | `DLIMG001` file | NCHW, shape from the file header | one-hot `[N, C]` |
| `CSVLoader` | `loaders/CSVLoader.hpp` | rectangular CSV | features `[N, F]` in `features()`, not inside a `Batch` | last columns in `targets()` |

`CSVLoader` uploads the whole table once. It has no `get_batch`. The caller slices the host copy.

## `ClassificationLoader`

The constructor scans `root/<split>/<class>/`. `class_names()` is the folder list in the order that defines the one-hot index. Pass the training loader's names into the test loader so a missing test folder does not shorten `C`.

`get_batch` prefetches the next host batch while the caller uses the current GPU batch.

## `PackedImageLoader`

`DLIMG001`, little-endian:

```text
bytes 0–7   magic "DLIMG001"
u32         n, channels, height, width, num_classes
u8          pixels[n * c * h * w]   NCHW
u8          labels[n]
```

The file is read into host memory once. Prefetch overlaps the conversion from `uint8` to float with the caller's GPU work. Images are scaled to `[0, 1]`.

## `CSVLoader`

Every non-empty row has the same number of columns. The last `target_columns` columns become the target tensor. `skip_header` drops the first row. Both tensors are rank 2.
