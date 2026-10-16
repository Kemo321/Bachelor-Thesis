"""Concatenate the project's own C++ into one file for full-context reading."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_DIRS = ("include", "src", "benchmarks", "tests")
HEADER_EXTENSIONS = {".h", ".hpp", ".cuh", ".inl"}
SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".cu"}
EXCLUDE_DIRS = {"build", "extern", "external", "_deps", ".git"}


def _skipped(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def _collect(extensions: set[str]) -> list[Path]:
    found: list[Path] = []
    for folder in CODE_DIRS:
        root = ROOT / folder
        if not root.is_dir():
            print(f"[WARNING] Folder {folder} does not exist. Skipping.")
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions and not _skipped(path):
                found.append(path)
    return sorted(found)


def create_combined_cpp(output_filename: str = "combined.cpp") -> None:
    out_path = (ROOT / output_filename).resolve()
    headers = [path for path in _collect(HEADER_EXTENSIONS) if path.resolve() != out_path]
    sources = [path for path in _collect(SOURCE_EXTENSIONS) if path.resolve() != out_path]

    print(f"Found {len(headers)} headers and {len(sources)} sources.")

    def write_section(outfile, files: list[Path], title: str) -> None:
        if not files:
            return
        outfile.write("// " + "=" * 60 + "\n")
        outfile.write(f"// SECTION: {title}\n")
        outfile.write("// " + "=" * 60 + "\n\n")
        for path in files:
            rel = path.resolve().relative_to(ROOT)
            outfile.write(f"// --- FILE: {rel.as_posix()} ---\n")
            outfile.write(path.read_text(encoding="utf-8", errors="replace"))
            outfile.write("\n\n")

    with out_path.open("w", encoding="utf-8", newline="\n") as outfile:
        outfile.write("// " + "=" * 60 + "\n")
        outfile.write("// AUTOMATICALLY GENERATED FILE - C++ PROJECT CONTEXT\n")
        outfile.write("// include, src, benchmarks, tests\n")
        outfile.write("// " + "=" * 60 + "\n\n")
        write_section(outfile, headers, "HEADER FILES")
        write_section(outfile, sources, "SOURCE FILES")

    print(f"Done! C++ context written to '{out_path.name}'.")


if __name__ == "__main__":
    create_combined_cpp()
