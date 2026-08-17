#!/usr/bin/env python3
"""Download and generate datasets for DeepLearnLib (VOC, BCCD, Synthetic3, CIFAR-10).

Existing target directories with data are skipped. Requires the Python standard
library plus Pillow or OpenCV to write JPEG images for Synthetic3.
"""

from __future__ import annotations

import argparse
import random
import shutil
import ssl
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable
from xml.etree.ElementTree import Element, ElementTree, SubElement

USER_AGENT = "DeepLearnLib-setup/1.0"
VOC_URLS = (
    "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
)
BCCD_ZIP_URL = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"
BCCD_GIT_URL = "https://github.com/Shenggan/BCCD_Dataset.git"
CIFAR10_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
CIFAR10_MIN_TRAIN_IMAGES = 50_000

SYNTH_CLASSES = ("square", "circle", "triangle")
SYNTH_IMAGE_SIZE = 448
SYNTH_COUNT = 180


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def log(message: str) -> None:
    print(f"[setup] {message}", flush=True)


BAR_WIDTH = 28


def _format_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MB"
    if n >= 1_000:
        return f"{n / 1e3:.0f} KB"
    return f"{n} B"


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--:--"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    def __init__(self, total: int, *, kind: str = "bytes", unit: str = "") -> None:
        self.total = max(total, 0)
        self.kind = kind
        self.unit = unit or ("files" if kind == "count" else "")
        self.start = time.monotonic()
        self._last_draw = 0.0

    def _label(self, current: int) -> str:
        if self.kind == "bytes":
            if self.total > 0:
                return f"{_format_bytes(current)}/{_format_bytes(self.total)}"
            return _format_bytes(current)
        if self.total > 0:
            return f"{current}/{self.total} {self.unit}"
        return f"{current} {self.unit}"

    def update(self, current: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_draw) < 0.08:
            return
        self._last_draw = now
        elapsed = max(now - self.start, 1e-6)
        speed = current / elapsed
        if self.total > 0:
            ratio = min(1.0, current / self.total)
            filled = int(BAR_WIDTH * ratio)
            bar = "#" * filled + "-" * (BAR_WIDTH - filled)
            percent = 100.0 * ratio
            remaining = ((self.total - current) / speed) if speed > 0 else 0.0
            extra = f"  {_format_eta(remaining)} left"
        else:
            bar = "#" * 4 + "-" * (BAR_WIDTH - 4)
            percent = 0.0
            extra = ""
        if self.kind == "bytes":
            rate = f"  {_format_bytes(int(speed))}/s"
        else:
            rate = f"  {speed:.0f} {self.unit}/s" if speed >= 1 else ""
        line = f"[setup]   [{bar}] {self._label(current)}  {percent:5.1f}%{rate}{extra}"
        print(f"\r{line}\033[K", end="", flush=True)

    def finish(self, current: int) -> None:
        self.update(current, force=True)
        print(flush=True)


def _ssl_context() -> ssl.SSLContext:
    try:
        return ssl.create_default_context()
    except ssl.SSLError:
        return ssl._create_unverified_context()


def download_file(urls: Iterable[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 1024:
        log(f"Already downloaded {destination.name} ({_format_bytes(destination.stat().st_size)})")
        return
    last_error: Exception | None = None
    for url in urls:
        log(f"Downloading {url}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, context=_ssl_context(), timeout=120) as response:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                bar = ProgressBar(total, kind="bytes")
                with destination.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        bar.update(downloaded)
                bar.finish(downloaded)
            return
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            print(flush=True)
            log(f"Failed {url}: {exc}")
            if destination.exists():
                destination.unlink()
    raise RuntimeError(f"Could not download {destination.name}") from last_error


def _tar_gzip(archive: Path) -> bool:
    return archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz"


def _extract_tar_via_system(archive: Path, dest: Path, gzip_compressed: bool) -> bool:
    tar_bin = shutil.which("tar")
    if tar_bin is None:
        return False
    flags = "-xz" if gzip_compressed else "-x"
    cmd = [tar_bin, flags, "-f", "-", "-C", str(dest)]
    if sys.platform != "win32":
        cmd.append("--no-same-owner")
    total = archive.stat().st_size
    bar = ProgressBar(total, kind="bytes")
    with archive.open("rb") as src:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        stderr_chunks: list[bytes] = []

        def drain_stderr() -> None:
            if proc.stderr is not None:
                stderr_chunks.append(proc.stderr.read())

        drain = threading.Thread(target=drain_stderr, daemon=True)
        drain.start()
        sent = 0
        try:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                proc.stdin.write(chunk)
                sent += len(chunk)
                bar.update(sent)
            proc.stdin.close()
            code = proc.wait()
            drain.join(timeout=5)
        except (BrokenPipeError, OSError) as exc:
            proc.kill()
            proc.wait()
            print(flush=True)
            log(f"system tar failed ({exc}); falling back to Python")
            return False
    if code != 0:
        message = b"".join(stderr_chunks).decode(errors="replace").strip()
        print(flush=True)
        log(f"system tar exited {code}: {message or 'no stderr'}; falling back to Python")
        return False
    bar.finish(sent)
    return True


def _tar_extract_kwargs() -> dict:
    if sys.version_info >= (3, 12):
        return {"filter": "data"}
    return {}


def _extract_tar_via_python(archive: Path, dest: Path, gzip_compressed: bool) -> None:
    mode = "r:gz" if gzip_compressed else "r:"
    total = archive.stat().st_size
    bar = ProgressBar(total, kind="bytes")
    kwargs = _tar_extract_kwargs()
    with tarfile.open(archive, mode) as tar:
        for member in tar:
            tar.extract(member, path=dest, **kwargs)
            pos = tar.fileobj.tell() if tar.fileobj is not None else 0
            underlying = getattr(tar.fileobj, "fileobj", None)
            if underlying is not None:
                pos = underlying.tell()
            bar.update(pos)
    bar.finish(total)


def _extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        bar = ProgressBar(len(names), kind="count", unit="files")
        for index, name in enumerate(names, start=1):
            zipped.extract(name, dest)
            bar.update(index)
        bar.finish(len(names))


def extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    size = _format_bytes(archive.stat().st_size)
    log(f"Extracting {archive.name} ({size}) -> {dest}")
    if _tar_gzip(archive):
        if not _extract_tar_via_system(archive, dest, gzip_compressed=True):
            _extract_tar_via_python(archive, dest, gzip_compressed=True)
        return
    if archive.suffix == ".tar":
        if not _extract_tar_via_system(archive, dest, gzip_compressed=False):
            _extract_tar_via_python(archive, dest, gzip_compressed=False)
        return
    if archive.suffix == ".zip":
        _extract_zip(archive, dest)
        return
    raise RuntimeError(f"Unsupported archive: {archive}")


def dir_has_files(path: Path, suffixes: tuple[str, ...] | None = None) -> bool:
    return count_files(path, suffixes) > 0


def count_files(path: Path, suffixes: tuple[str, ...] | None = None) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        if suffixes is None or child.suffix.lower() in suffixes:
            total += 1
    return total


def _try_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        return cv2, np
    except ImportError:
        return None, None


def _try_pil():
    try:
        from PIL import Image, ImageDraw  # type: ignore

        return Image, ImageDraw
    except ImportError:
        return None, None


def make_canvas(size: int, rng: random.Random):
    bg = tuple(rng.randint(20, 60) for _ in range(3))
    cv2, np = _try_cv2()
    if np is not None:
        return np.full((size, size, 3), bg[::-1], dtype=np.uint8)

    Image, _ = _try_pil()
    if Image is None:
        raise RuntimeError("Synthetic3 needs OpenCV (cv2) or Pillow")
    return Image.new("RGB", (size, size), bg)


def draw_shape(canvas, name: str, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    xmin, ymin, xmax, ymax = box
    cv2, np = _try_cv2()
    if cv2 is not None and np is not None:
        bgr = color[::-1]
        if name == "square":
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), bgr, thickness=-1)
        elif name == "circle":
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            radius = max(1, min(xmax - xmin, ymax - ymin) // 2)
            cv2.circle(canvas, (cx, cy), radius, bgr, thickness=-1)
        else:
            pts = np.array(
                [[[(xmin + xmax) // 2, ymin], [xmin, ymax], [xmax, ymax]]],
                dtype=np.int32,
            )
            cv2.fillPoly(canvas, pts, bgr)
        return

    ImageDraw = _try_pil()[1]
    draw = ImageDraw.Draw(canvas)
    if name == "square":
        draw.rectangle([xmin, ymin, xmax, ymax], fill=color)
    elif name == "circle":
        draw.ellipse([xmin, ymin, xmax, ymax], fill=color)
    else:
        draw.polygon([((xmin + xmax) // 2, ymin), (xmin, ymax), (xmax, ymax)], fill=color)


def write_canvas(path: Path, canvas) -> None:
    cv2, np = _try_cv2()
    if cv2 is not None and np is not None:
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError(f"cv2.imwrite failed: {path}")
        return
    canvas.save(path, format="JPEG", quality=92)


def write_voc_xml(path: Path, filename: str, size: int, objects: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    root = Element("annotation")
    SubElement(root, "folder").text = "JPEGImages"
    SubElement(root, "filename").text = filename
    size_node = SubElement(root, "size")
    SubElement(size_node, "width").text = str(size)
    SubElement(size_node, "height").text = str(size)
    SubElement(size_node, "depth").text = "3"
    for name, (xmin, ymin, xmax, ymax) in objects:
        obj = SubElement(root, "object")
        SubElement(obj, "name").text = name
        SubElement(obj, "pose").text = "Unspecified"
        SubElement(obj, "truncated").text = "0"
        SubElement(obj, "difficult").text = "0"
        box = SubElement(obj, "bndbox")
        SubElement(box, "xmin").text = str(xmin)
        SubElement(box, "ymin").text = str(ymin)
        SubElement(box, "xmax").text = str(xmax)
        SubElement(box, "ymax").text = str(ymax)
    ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def setup_voc(data_root: Path) -> None:
    jpeg = data_root / "VOCdevkit" / "VOC2012" / "JPEGImages"
    if dir_has_files(jpeg, (".jpg", ".jpeg", ".png")):
        log("VOC 2012 already present; skipping download")
        return
    archive = data_root / "_downloads" / "VOCtrainval_11-May-2012.tar"
    download_file(VOC_URLS, archive)
    extract_archive(archive, data_root)
    if not dir_has_files(jpeg, (".jpg", ".jpeg", ".png")):
        raise RuntimeError(f"VOC extract did not produce {jpeg}")
    log(f"VOC 2012 ready at {data_root / 'VOCdevkit'}")


def setup_bccd(data_root: Path) -> None:
    jpeg = data_root / "BCCD_Dataset" / "BCCD" / "JPEGImages"
    if dir_has_files(jpeg, (".jpg", ".jpeg", ".png")):
        log("BCCD already present; skipping download")
        return

    dest = data_root / "BCCD_Dataset"
    archive = data_root / "_downloads" / "BCCD_Dataset.zip"
    try:
        download_file((BCCD_ZIP_URL,), archive)
        extract_dir = data_root / "_downloads" / "bccd_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_archive(archive, extract_dir)
        candidates = list(extract_dir.glob("BCCD_Dataset-*"))
        source = candidates[0] if candidates else extract_dir
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(source), str(dest))
    except Exception as exc:
        log(f"Zip download failed ({exc}); trying git clone")
        if dest.exists():
            shutil.rmtree(dest)
        subprocess.run(["git", "clone", "--depth", "1", BCCD_GIT_URL, str(dest)], check=True)

    if not dir_has_files(jpeg, (".jpg", ".jpeg", ".png")):
        raise RuntimeError(f"BCCD extract did not produce {jpeg}")
    log(f"BCCD ready at {dest}")


def setup_synthetic3(data_root: Path) -> None:
    train_root = data_root / "Synthetic3" / "train"
    jpeg_dir = train_root / "JPEGImages"
    annot_dir = train_root / "Annotations"
    if dir_has_files(jpeg_dir, (".jpg", ".jpeg")):
        log("Synthetic3 already present; skipping generation")
        return
    if _try_cv2()[0] is None and _try_pil()[0] is None:
        raise RuntimeError("Synthetic3 generation requires Pillow or opencv-python")

    jpeg_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    log(f"Generating {SYNTH_COUNT} Synthetic3 images ({SYNTH_IMAGE_SIZE}x{SYNTH_IMAGE_SIZE})")
    bar = ProgressBar(SYNTH_COUNT, kind="count", unit="images")
    for index in range(SYNTH_COUNT):
        filename = f"{index:06d}.jpg"
        canvas = make_canvas(SYNTH_IMAGE_SIZE, rng)
        objects: list[tuple[str, tuple[int, int, int, int]]] = []
        for _ in range(rng.randint(1, 3)):
            name = rng.choice(SYNTH_CLASSES)
            side = rng.randint(40, 140)
            xmin = rng.randint(8, SYNTH_IMAGE_SIZE - side - 8)
            ymin = rng.randint(8, SYNTH_IMAGE_SIZE - side - 8)
            xmax = xmin + side
            ymax = ymin + side
            color = tuple(rng.randint(80, 255) for _ in range(3))
            draw_shape(canvas, name, (xmin, ymin, xmax, ymax), color)
            objects.append((name, (xmin, ymin, xmax, ymax)))
        write_canvas(jpeg_dir / filename, canvas)
        write_voc_xml(annot_dir / f"{index:06d}.xml", filename, SYNTH_IMAGE_SIZE, objects)
        bar.update(index + 1)
    bar.finish(SYNTH_COUNT)
    log(f"Synthetic3 ready at {train_root}")


def setup_cifar10(data_root: Path) -> None:
    train_dir = data_root / "cifar10" / "train"
    image_suffixes = (".jpg", ".jpeg", ".png")
    existing = count_files(train_dir, image_suffixes)
    if existing >= CIFAR10_MIN_TRAIN_IMAGES:
        log("CIFAR-10 already present; skipping download")
        return
    if existing > 0:
        log(f"CIFAR-10 incomplete ({existing} train images); extracting again")
        shutil.rmtree(data_root / "cifar10", ignore_errors=True)
    archive = data_root / "_downloads" / "cifar10.tgz"
    download_file((CIFAR10_URL,), archive)
    extract_archive(archive, data_root)
    extracted = count_files(train_dir, image_suffixes)
    if extracted < CIFAR10_MIN_TRAIN_IMAGES:
        raise RuntimeError(f"CIFAR-10 extract did not produce {train_dir}")
    log(f"CIFAR-10 ready at {data_root / 'cifar10'} ({extracted} train images)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and generate DeepLearnLib datasets.")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root (default: <repo>/data)")
    parser.add_argument("--only", choices=("voc", "bccd", "synthetic", "cifar10"), action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = (args.data_root or (repo_root() / "data")).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Callable[[Path], None]]] = [
        ("voc", setup_voc),
        ("bccd", setup_bccd),
        ("synthetic", setup_synthetic3),
        ("cifar10", setup_cifar10),
    ]
    selected = set(args.only or [name for name, _ in jobs])
    try:
        for name, func in jobs:
            if name in selected:
                func(data_root)
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1
    log("All requested datasets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
