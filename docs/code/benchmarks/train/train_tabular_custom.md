# train_tabular_custom.cpp

Trains `TabularMLP`. The pipeline name is the first argument (`tabular_demo` when it is omitted). The epoch driver is `train_custom_rows`, shared with the GRU program.

## main

1. Loads the CSV. A missing file is replaced by the synthetic table from `write_dummy_csv` (seed 42).
2. Holds out `test_fraction` of the rows with seed 42.
3. Shuffles the training indices with a second generator, also seed 42, once per epoch.
4. Writes `metrics_custom.csv` (train and test columns) and `tabular_mlp_final.bin`.
