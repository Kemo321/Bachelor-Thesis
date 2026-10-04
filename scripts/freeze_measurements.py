#!/usr/bin/env python3
"""Copy the last row of each results/**/metrics_*.csv into docs/usage/measurements.md.

results/ is gitignored. This script does not invent numbers. With no CSV files
it writes a page that says the table is empty. GitHub Actions has no GPU and
does not produce these files.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "usage" / "measurements.md"


def metric_files() -> list[Path]:
    if not RESULTS.is_dir():
        return []
    return sorted(path for path in RESULTS.rglob("metrics_*.csv") if path.is_file())


def last_row(path: Path) -> tuple[list[str], list[str]] | None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    if len(rows) < 2:
        return None
    header = rows[0]
    data = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    if not data:
        return None
    return header, data[-1]


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    def cell(value: str) -> str:
        return value.replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(name) for name in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(cell(value) for value in padded[: len(header)]) + " |")
    return "\n".join(lines)


def render(files: list[Path]) -> str:
    intro = """# Measurements

`results/` is gitignored. This page is the copy that can go into the thesis. Refresh it after a local GPU run:

```bash
python scripts/freeze_measurements.py
```

GitHub-hosted runners have no NVIDIA GPU. CI compiles the training binaries and does not execute them, so this table stays empty until a machine with a GPU has written `results/**/metrics_*.csv`. The script copies the last row of each file. It does not fill in missing runs.
"""
    if not files:
        body = """
## Recorded runs

No `results/**/metrics_*.csv` is in the tree. There is nothing to copy.
"""
        return intro + body

    sections = ["\n## Recorded runs\n"]
    for path in files:
        parsed = last_row(path)
        relative = path.relative_to(ROOT).as_posix()
        sections.append(f"### `{relative}`\n")
        if parsed is None:
            sections.append("The file has a header and no data row.\n")
            continue
        header, row = parsed
        sections.append(markdown_table(["File", *header], [[relative, *row]]))
        sections.append("")
    return intro + "\n".join(sections) + "\n"


def main() -> None:
    text = render(metric_files())
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
