#!/usr/bin/env python3
"""Write reports/coverage.md from the test suite and, when present, an lcov tracefile.

File coverage counts a library, model, or shared benchmark header as covered when a
test mentions its name. Line rates are copied from lcov when --lcov is passed.
Training drivers are listed apart from that percentage: they are programs, and the
loop they share is what the tests lock down.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

LIBRARY_ROOTS = (
    ROOT / "src",
    ROOT / "include" / "DeepLearnLib",
    ROOT / "benchmarks" / "models",
    ROOT / "benchmarks" / "common",
)
DRIVER_ROOTS = (
    ROOT / "benchmarks" / "train",
    ROOT / "benchmarks" / "forward",
    ROOT / "benchmarks" / "micro",
)
SOURCE_SUFFIXES = {".cpp", ".cu", ".hpp", ".h", ".cuh"}


def iter_sources(roots: tuple[Path, ...]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            found.append(path)
    return sorted(found)


def test_texts() -> list[tuple[Path, str]]:
    tests = ROOT / "tests"
    rows: list[tuple[Path, str]] = []
    if not tests.is_dir():
        return rows
    for path in sorted(tests.rglob("*")):
        if path.is_file() and path.suffix in {".cpp", ".hpp"}:
            rows.append((path, path.read_text(encoding="utf-8", errors="replace")))
    return rows


def matching_tests(path: Path, tests: list[tuple[Path, str]]) -> list[str]:
    stem = path.stem
    tokens = {stem, stem.lower(), stem.replace("_", "")}
    if stem.startswith("test_"):
        tokens.add(stem[len("test_") :])
    hits: list[str] = []
    for test_path, text in tests:
        haystack = test_path.stem + "\n" + text
        if any(token and len(token) >= 3 and token in haystack for token in tokens):
            hits.append(test_path.relative_to(ROOT).as_posix())
    return hits


def parse_lcov(path: Path) -> dict[str, dict[str, int]]:
    rates: dict[str, dict[str, int]] = {}
    current: str | None = None
    found = 0
    hit = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("SF:"):
            current = line[3:].replace("\\", "/")
            found = 0
            hit = 0
        elif line.startswith("LF:"):
            found = int(line[3:])
        elif line.startswith("LH:"):
            hit = int(line[3:])
        elif line == "end_of_record" and current:
            rates[current] = {"lines_found": found, "lines_hit": hit}
            current = None
    return rates


def lcov_for(path: Path, rates: dict[str, dict[str, int]]) -> dict[str, int] | None:
    suffix = path.relative_to(ROOT).as_posix()
    for key, value in rates.items():
        if key.endswith(suffix):
            return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the DeepLearnLib coverage report.")
    parser.add_argument("--lcov", type=Path, default=None, help="Optional lcov .info tracefile.")
    parser.add_argument("--output", type=Path, default=REPORTS / "coverage.md")
    args = parser.parse_args()

    tests = test_texts()
    library = iter_sources(LIBRARY_ROOTS)
    drivers = iter_sources(DRIVER_ROOTS)
    rates = parse_lcov(args.lcov) if args.lcov and args.lcov.is_file() else {}

    rows = []
    for path in library:
        hits = matching_tests(path, tests)
        line = lcov_for(path, rates)
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "tested": bool(hits),
                "tests": hits,
                "lines_found": None if line is None else line["lines_found"],
                "lines_hit": None if line is None else line["lines_hit"],
            }
        )

    covered = sum(1 for row in rows if row["tested"])
    total = len(rows)
    percent = (100.0 * covered / total) if total else 0.0
    line_found = sum(row["lines_found"] or 0 for row in rows)
    line_hit = sum(row["lines_hit"] or 0 for row in rows)
    line_percent = (100.0 * line_hit / line_found) if line_found else None

    driver_rows = []
    for path in drivers:
        driver_rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "tests": matching_tests(path, tests),
            }
        )

    payload = {
        "file_coverage_percent": round(percent, 2),
        "files_covered": covered,
        "files_total": total,
        "line_coverage_percent": None if line_percent is None else round(line_percent, 2),
        "lines_hit": line_hit if line_found else None,
        "lines_found": line_found if line_found else None,
        "lcov": None if not rates else str(args.lcov),
        "files": rows,
        "drivers": driver_rows,
    }

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Coverage",
        "",
        "This report maps library sources, application models, and the shared benchmark headers onto the GoogleTest suite.",
        "A file counts as covered when a test names it. Training and forward programs are drivers: they are not part of that percentage, because the loop, the CSV, and the models they call are what the tests check.",
        "",
        f"- Files covered: **{covered} / {total}** ({percent:.1f}%)",
    ]
    if line_percent is None:
        lines.append("- Line coverage: not in this file. Run with `--lcov coverage.filtered.info` after a `--coverage` build. CI does that on the debug job.")
    else:
        lines.append(f"- Line coverage from lcov: **{line_hit} / {line_found}** ({line_percent:.1f}%)")
    lines.extend(["", "## Library, models, and shared headers", "", "| File | Tests | Lines |", "| --- | --- | --- |"])
    for row in rows:
        test_cell = ", ".join(f"`{name}`" for name in row["tests"]) if row["tests"] else "—"
        if row["lines_found"]:
            line_cell = f"{row['lines_hit']} / {row['lines_found']}"
        else:
            line_cell = "—"
        lines.append(f"| `{row['path']}` | {test_cell} | {line_cell} |")

    missing = [row["path"] for row in rows if not row["tested"]]
    lines.extend(["", "## Files with no matching test", ""])
    if missing:
        lines.extend(f"- `{path}`" for path in missing)
    else:
        lines.append("Every listed file is named by at least one test.")

    lines.extend(["", "## Training and benchmark drivers", "", "These programs are the paired custom and Torch entry points. They are compiled by the release job and are not unit tests.", ""])
    for row in driver_rows:
        note = ", ".join(f"`{name}`" for name in row["tests"]) if row["tests"] else "paired binary"
        lines.append(f"- `{row['path']}` — {note}")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    try:
        shown = output.relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(output)
    print(f"Wrote {shown} ({covered}/{total} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
