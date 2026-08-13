import re
import subprocess
import sys
from pathlib import Path


def exports_from_dll(dumpbin: str, dll: Path) -> list[str]:
    result = subprocess.run([dumpbin, "/exports", str(dll)], capture_output=True, text=True, check=True)
    names: list[str] = []
    in_table = False
    for line in result.stdout.splitlines():
        if "ordinal hint" in line.lower() and "name" in line.lower():
            in_table = True
            continue
        if in_table:
            if not line.strip():
                if names:
                    break
                continue
            match = re.match(r"\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)", line)
            if match:
                names.append(match.group(1))
    return names


def write_import_lib(lib_exe: str, dll: Path, out_dir: Path) -> Path:
    names = exports_from_dll("dumpbin", dll)
    if not names:
        raise RuntimeError(f"No exports found in {dll}")
    def_path = out_dir / (dll.stem + ".def")
    lib_path = out_dir / (dll.stem.replace("64_9", "") + ".lib")
    # cudnn64_9.dll -> cudnn.lib ; cudnn_ops64_9.dll -> cudnn_ops.lib
    stem = dll.stem
    if stem.endswith("64_9"):
        lib_path = out_dir / (stem[: -len("64_9")] + ".lib")
    def_path.write_text("LIBRARY " + dll.name + "\nEXPORTS\n" + "\n".join(names) + "\n", encoding="utf-8")
    subprocess.run([lib_exe, f"/def:{def_path}", f"/out:{lib_path}", "/machine:x64"], check=True)
    return lib_path


def main() -> None:
    dll_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for dll in sorted(dll_dir.glob("cudnn*.dll")):
        path = write_import_lib("lib", dll, out_dir)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
