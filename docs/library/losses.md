# Losses

Loss types are not layers. They do not hold parameters and they do not implement `step`. Each one has a static `loss` and a static `loss_derivative`. Both run on GPU tensors and return GPU tensors. Reading the scalar requires `to_host` at the call site.

## `YOLOLoss`

Grid size `S = 7`, two boxes per cell, `C` classes. The prediction is `[N, 7, 7, 10 + C]` or the flat `[N, 7*7*(10+C)]` produced by a linear head. Per cell:

| Offset | Meaning |
| --- | --- |
| 0–3 | box 1: `x`, `y` relative to the cell, `w`, `h` relative to the image |
| 4 | box 1 objectness |
| 5–8 | box 2, same layout |
| 9 | box 2 objectness |
| 10 … | class scores |

One device thread handles one cell. Constants in the kernel are `lambda_coord = 5` and `lambda_noobj = 0.5`.

For each cell the kernel:

1. Decodes both predicted boxes and the target box into image coordinates and computes IoU.
2. Marks the predictor with the higher IoU as responsible, and only when the target objectness is 1.
3. Adds coordinate loss on that predictor: squared error on `x` and `y`, and on `sqrt(w)` and `sqrt(h)`, multiplied by `lambda_coord`.
4. Adds objectness loss: squared error against that IoU for the responsible predictor, plus `lambda_noobj` times the squared confidence of predictors that are not responsible.
5. Adds class loss: squared error on the class vector, multiplied by target objectness.

A second kernel sums the per-cell values and multiplies by `1 / N`. The scalar is the sum of cell losses averaged over the batch, not the mean of every cell. `loss_derivative` writes `dL/dpred` in one kernel with the same `1 / N` factor.

The target tensor uses the same layout. `CustomDataLoader` fills it; the encoding is documented in [Loaders](loaders.md).

## `CrossEntropyLoss`

Mean softmax cross-entropy on logits `[N, C]` and a dense target of the same shape. The shipped loaders pass a one-hot target. Softmax is inside both `loss` and `loss_derivative`. The gradient is `softmax(logits) - target`, averaged over the batch. Callers pass logits.

## `MSELoss`

Mean squared error over all elements, and the matching derivative. It has no dataset-specific layout.
