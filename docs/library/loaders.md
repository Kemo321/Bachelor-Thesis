# Loaders

A `Batch` is two GPU tensors: `images` and `targets`. Loaders own host decode and the upload. They do not run `forward` or `backward`.

| Class | Header | Source | `images` | `targets` |
| --- | --- | --- | --- | --- |
| `CustomDataLoader` | `dataset.hpp` | image paths and VOC-style XML | NCHW `[N, 3, 448, 448]`, float in `[0, 1]` | YOLO grid, width `10 + num_classes` |
| `ClassificationLoader` | `ClassificationLoader.hpp` | `root/<split>/<class>/*.jpg` | NCHW, side length from the constructor | one-hot `[N, C]` |
| `PackedImageLoader` | `PackedImageLoader.hpp` | `DLIMG001` file | NCHW, shape from the file header | one-hot `[N, C]` |
| `CSVLoader` | `CSVLoader.hpp` | rectangular CSV | features `[N, F]` in `features()`, not inside a `Batch` | last columns in `targets()` |

`CSVLoader` uploads the whole table once. It has no `get_batch`. The caller slices the host copy.

## `CustomDataLoader`

`split_dataset` fills three `DataPaths` (image paths and annotation paths) from a VOC-style root. Default ratios are 0.7 / 0.15 / 0.15. The default class list is the 20 VOC names. Any other list sets `C` for the grid.

For each ground-truth box the loader picks the cell from the box centre and writes the first predictor slot whose objectness is still 0:

- `x`, `y` — offset of the centre inside the cell;
- `w`, `h` — box size as a fraction of the image;
- objectness — `1`;
- classes — one-hot starting at offset 10.

A third box in the same cell is dropped. There are only two slots.

When `is_train` is true, the loader applies affine scale and translation and HSV jitter on the CPU after decode. Eval does not.

`reset` restarts the cursor and, in training, shuffles the index order. `has_next` / `get_batch(stream)` walk that order. `get_batch` uploads on `stream` and starts a `std::future` that decodes the following host batch. The next `get_batch` waits for that future. Decode of one batch uses `dl::parallel_for`: `min(batch, hardware_concurrency, 16)` workers, strided, not one thread per image.

Passing a non-default stream lets the caller overlap the upload with compute on another stream. The loader does not create that second stream. Applications that do are described in [Usage](../usage/training.md).

## `ClassificationLoader`

The constructor scans `root/<split>/<class>/`. `class_names()` is the folder list in the order that defines the one-hot index. Pass the training loader's names into the test loader so a missing test folder does not shorten `C`.

`get_batch` has the same prefetch contract as `CustomDataLoader`.

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
