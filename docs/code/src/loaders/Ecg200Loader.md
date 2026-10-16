# Ecg200Loader.cpp

Reads one UCR ECG200 split into a flat series buffer and integer labels. The file may be the classic label-then-samples row, or a `series:label` line from a `.ts` file. Lines that are empty or start with `#` or `@` are skipped.

## trim

Drops leading and trailing whitespace.

## split_char

Splits on one separator and trims each piece.

## parse_floats

Reads whitespace-separated floats from a line.

## parse_row

Returns nothing for a comment or an empty line. A colon splits a `.ts` row into the series and the label, in either order. A row without a colon treats the first value as the label and the rest as samples. A row with no samples throws.

## label_name

Prints an integer label without a decimal, and any other label with `std::to_string`.

## load_ecg200_file

Parses every data row, checks that the series length is constant, and maps sorted distinct labels onto `0 .. C-1`. The values are stored row-major `[N, T]`. Fewer than two classes throws.

## find_ecg200_split

Looks for `ECG200_<which>.txt`, `.tsv`, and `.ts` directly under the root and under `root/ECG200`. The error names the archive to download when none of those files exist.
