# CSVLoader.cpp

Loads a rectangular numeric CSV and, in the constructor, uploads the whole table to the GPU through `from_host`. Features are every column except the last `target_columns`. Targets are those last columns. The tensors are rank 2: features `[N, F]`, targets `[N, T]`. The default arguments in the declaration are `target_columns = 1` and `skip_header = false`.

## parse_csv

Rejects `target_columns <= 0` and a file that cannot be opened. When `skip_header` is set, the first line must exist and is skipped, not parsed. Empty lines and a line equal to `"\r"` alone are dropped. A trailing `\r` left by `getline` is stripped, because `stof` will not accept it at the end of a cell.

Every cell must be non-empty and parse as a float. Otherwise the exception carries the cell text. A row with no values is dropped. The column count comes from the first data row. A different row length throws a rectangular-CSV error. No data rows, and a column count that is not greater than `target_columns`, also throw.

The result splits each row into two flat vectors: the leading columns go to features, and the last `target_columns` go to targets.

## CSVLoader::CSVLoader(dl::Tensor, dl::Tensor)

Private constructor. Stores the feature and target tensors passed in after `from_host`. `from_parsed` calls it.

## CSVLoader::from_parsed

Calls `parse_csv`, then `from_host` twice on the GPU (`float32`, the default stream), and builds the loader from those tensors. The whole table goes to the device at once, not split into successive batches.

## CSVLoader::CSVLoader(std::string, int, bool)

Public constructor. Delegates to the object returned by `from_parsed`, so parsing and the upload are finished when construction ends.

## CSVLoader::features

Returns a reference to the feature tensor `[N, F]`.

## CSVLoader::targets

Returns a reference to the target tensor `[N, T]` from the last columns of the file.

## CSVLoader::size

Returns the first dimension of the feature tensor, which is the row count set in `from_parsed`.
