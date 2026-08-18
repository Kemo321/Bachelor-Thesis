#!/usr/bin/env python3
"""Generate thesis-quality PNG figures from pipeline metrics CSVs.

Looks for ``metrics_custom.csv`` and ``metrics_torch.csv`` under each experiment
directory (VOC, BCCD, synthetic, CIFAR-10, MNIST, tabular variants, overfit).
Also plots accuracy, confusion matrices, and classification sample grids when
those artifacts are present. Missing Torch files or mAP columns are skipped
with a warning rather than aborting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "plot_metrics.py requires matplotlib and pandas.\n"
        "Install with: pip install matplotlib pandas\n"
        f"Import error: {exc}\n"
    )
    raise SystemExit(1) from exc

EXPERIMENT_DIRS = (
    "voc",
    "bccd",
    "synthetic",
    "cifar10",
    "mnist",
    "tabular",
    "tabular_iris",
    "tabular_wisconsin",
    "overfit",
    "voc_short",
)

CUSTOM_TRAIN = "#1B4F72"
CUSTOM_TEST = "#5DADE2"
TORCH_TRAIN = "#922B21"
TORCH_TEST = "#E59866"
CUSTOM_MAP = "#117A65"
TORCH_MAP = "#B9770E"
CUSTOM_TIME = "#1F618D"
TORCH_TIME = "#A04000"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#D5D8DC",
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "")
        if key.startswith("epoch"):
            rename[column] = "Epoch"
        elif key in {"acc", "trainacc"} or ("train" in key and "acc" in key):
            rename[column] = "TrainAcc"
        elif key in {"testacc", "valacc"} or (("test" in key or "val" in key) and "acc" in key):
            rename[column] = "TestAcc"
        elif key in {"loss", "trainloss"} or ("train" in key and "loss" in key):
            rename[column] = "TrainLoss"
        elif key in {"testloss", "valloss"} or (("test" in key or "val" in key) and "loss" in key):
            rename[column] = "TestLoss"
        elif "map" in key:
            rename[column] = "mAP@0.5"
        elif "time" in key:
            rename[column] = "Time(s)"
        elif "vram" in key:
            rename[column] = "VRAM_MiB"
    return frame.rename(columns=rename)


def load_metrics(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, sep=";")
        if frame.shape[1] == 1:
            frame = pd.read_csv(path)
        frame = _normalize_columns(frame)
        if "Epoch" not in frame.columns:
            print(f"[plot] Skipping {path}: no Epoch column", file=sys.stderr)
            return None
        frame = frame.sort_values("Epoch")
        return frame
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"[plot] Failed to parse {path}: {exc}", file=sys.stderr)
        return None


def _save(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    print(f"[plot] Wrote {destination}")


def plot_train_vs_test_loss(
    custom: pd.DataFrame | None, torch_df: pd.DataFrame | None, title: str, destination: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    plotted = False
    if custom is not None and "TrainLoss" in custom.columns:
        ax.plot(custom["Epoch"], custom["TrainLoss"], color=CUSTOM_TRAIN, lw=1.8, label="Custom train")
        plotted = True
        if "TestLoss" in custom.columns:
            ax.plot(custom["Epoch"], custom["TestLoss"], color=CUSTOM_TEST, lw=1.8, ls="--", label="Custom test")
    if torch_df is not None and "TrainLoss" in torch_df.columns:
        ax.plot(torch_df["Epoch"], torch_df["TrainLoss"], color=TORCH_TRAIN, lw=1.8, label="Torch train")
        plotted = True
        if "TestLoss" in torch_df.columns:
            ax.plot(torch_df["Epoch"], torch_df["TestLoss"], color=TORCH_TEST, lw=1.8, ls="--", label="Torch test")
    if not plotted:
        plt.close(fig)
        print(f"[plot] No loss columns for {title}; skipping train/test figure.")
        return
    ax.set_title(f"{title}: Train vs Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    _save(fig, destination)


def plot_map(
    custom: pd.DataFrame | None, torch_df: pd.DataFrame | None, title: str, destination: Path
) -> None:
    series: list[tuple[str, pd.DataFrame, str]] = []
    if custom is not None and "mAP@0.5" in custom.columns:
        series.append(("Custom mAP@0.5", custom, CUSTOM_MAP))
    if torch_df is not None and "mAP@0.5" in torch_df.columns:
        series.append(("Torch mAP@0.5", torch_df, TORCH_MAP))
    if not series:
        print(f"[plot] No mAP@0.5 column for {title}; skipping mAP figure.")
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for label, frame, color in series:
        ax.plot(frame["Epoch"], frame["mAP@0.5"], color=color, lw=1.9, marker="o", ms=3.2, markevery=max(1, len(frame) // 20), label=label)
    ax.set_title(f"{title}: mAP@0.5 over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    _save(fig, destination)


def plot_epoch_duration(
    custom: pd.DataFrame | None, torch_df: pd.DataFrame | None, title: str, destination: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    plotted = False
    if custom is not None and "Time(s)" in custom.columns:
        ax.plot(custom["Epoch"], custom["Time(s)"], color=CUSTOM_TIME, lw=1.8, label="Custom")
        plotted = True
    if torch_df is not None and "Time(s)" in torch_df.columns:
        ax.plot(torch_df["Epoch"], torch_df["Time(s)"], color=TORCH_TIME, lw=1.8, ls="--", label="Torch")
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"[plot] No duration column for {title}; skipping duration figure.")
        return
    ax.set_title(f"{title}: Epoch Duration Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Duration (s)")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    _save(fig, destination)


def plot_vram(
    custom: pd.DataFrame | None, torch_df: pd.DataFrame | None, title: str, destination: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    plotted = False
    if custom is not None and "VRAM_MiB" in custom.columns:
        ax.plot(custom["Epoch"], custom["VRAM_MiB"], color=CUSTOM_TIME, lw=1.8, label="Custom")
        plotted = True
    if torch_df is not None and "VRAM_MiB" in torch_df.columns:
        ax.plot(torch_df["Epoch"], torch_df["VRAM_MiB"], color=TORCH_TIME, lw=1.8, ls="--", label="Torch")
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"[plot] No VRAM_MiB column for {title}; skipping VRAM figure.")
        return
    ax.set_title(f"{title}: VRAM Usage")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VRAM (MiB)")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    _save(fig, destination)


def plot_accuracy(
    custom: pd.DataFrame | None, torch_df: pd.DataFrame | None, title: str, destination: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    plotted = False
    if custom is not None and "TrainAcc" in custom.columns:
        ax.plot(custom["Epoch"], custom["TrainAcc"], color=CUSTOM_TRAIN, lw=1.8, label="Custom train acc")
        plotted = True
        if "TestAcc" in custom.columns:
            ax.plot(custom["Epoch"], custom["TestAcc"], color=CUSTOM_TEST, lw=1.8, ls="--", label="Custom test acc")
    if torch_df is not None and "TrainAcc" in torch_df.columns:
        ax.plot(torch_df["Epoch"], torch_df["TrainAcc"], color=TORCH_TRAIN, lw=1.8, label="Torch train acc")
        plotted = True
        if "TestAcc" in torch_df.columns:
            ax.plot(torch_df["Epoch"], torch_df["TestAcc"], color=TORCH_TEST, lw=1.8, ls="--", label="Torch test acc")
    if not plotted:
        plt.close(fig)
        print(f"[plot] No accuracy columns for {title}; skipping accuracy figure.")
        return
    ax.set_title(f"{title}: Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    _save(fig, destination)


def plot_confusion(path: Path, title: str, destination: Path) -> None:
    if not path.is_file():
        return
    try:
        frame = pd.read_csv(path, sep=";")
        if frame.shape[1] == 1:
            frame = pd.read_csv(path)
        frame = frame.set_index(frame.columns[0])
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"[plot] Failed to parse confusion {path}: {exc}", file=sys.stderr)
        return
    values = frame.to_numpy(dtype=float)
    if values.size == 0:
        return
    labels = [str(column) for column in frame.columns]
    fig_w = max(5.5, min(12.0, 0.55 * len(labels) + 3.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.86))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    max_value = float(values.max()) if values.size else 0.0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if values[row, col] > (0.55 * max_value) else "#1B2631"
            ax.text(col, row, f"{int(values[row, col])}", ha="center", va="center", color=color, fontsize=8)
    _save(fig, destination)


def plot_sample_grid(directory: Path, title: str, destination: Path) -> None:
    if not directory.is_dir():
        return
    images = sorted(path for path in directory.glob("*.png"))[:16]
    if not images:
        return
    cols = 4
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for index, axis in enumerate(axes_list):
        axis.axis("off")
        if index >= len(images):
            continue
        axis.imshow(plt.imread(str(images[index])))
        axis.set_title(images[index].stem.replace("_", " "), fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, destination)


def experiment_title(name: str) -> str:
    mapping = {
        "voc": "PASCAL VOC",
        "bccd": "BCCD",
        "synthetic": "Synthetic",
        "cifar10": "CIFAR-10",
        "mnist": "MNIST",
        "tabular": "Tabular demo",
        "tabular_iris": "Iris",
        "tabular_wisconsin": "Wisconsin WDBC",
        "overfit": "VOC Overfit",
        "voc_short": "VOC Short",
    }
    return mapping.get(name.lower(), name.replace("_", " ").title())


def discover_experiments(results_root: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for name in EXPERIMENT_DIRS:
        path = results_root / name
        if path.is_dir():
            found.append(path)
            seen.add(path.resolve())
    if results_root.is_dir():
        for child in sorted(results_root.iterdir()):
            if not child.is_dir() or child.name == "plots":
                continue
            resolved = child.resolve()
            if resolved in seen:
                continue
            if (child / "metrics_custom.csv").is_file() or (child / "metrics_torch.csv").is_file():
                found.append(child)
                seen.add(resolved)
    return found


def process_experiment(experiment_dir: Path, gallery_dir: Path) -> int:
    custom = load_metrics(experiment_dir / "metrics_custom.csv")
    torch_df = load_metrics(experiment_dir / "metrics_torch.csv")
    if custom is None and torch_df is None:
        print(f"[plot] No metrics CSVs in {experiment_dir}")
        return 0
    if torch_df is None:
        print(f"[plot] Torch CSV missing in {experiment_dir}; plotting Custom only.")

    title = experiment_title(experiment_dir.name)
    plot_dir = experiment_dir / "plots"
    written = 0
    outputs = [
        ("train_vs_test_loss.png", plot_train_vs_test_loss),
        ("accuracy.png", plot_accuracy),
        ("map50.png", plot_map),
        ("epoch_duration.png", plot_epoch_duration),
        ("vram.png", plot_vram),
    ]
    for filename, plotter in outputs:
        dest = plot_dir / filename
        plotter(custom, torch_df, title, dest)
        if dest.is_file():
            gallery = gallery_dir / f"{experiment_dir.name}_{filename}"
            gallery.parent.mkdir(parents=True, exist_ok=True)
            gallery.write_bytes(dest.read_bytes())
            written += 1

    extra = [
        (experiment_dir / "confusion_custom.csv", f"{title}: Custom confusion", "confusion_custom.png"),
        (experiment_dir / "confusion_torch.csv", f"{title}: Torch confusion", "confusion_torch.png"),
    ]
    for csv_path, conf_title, filename in extra:
        dest = plot_dir / filename
        plot_confusion(csv_path, conf_title, dest)
        if dest.is_file():
            gallery = gallery_dir / f"{experiment_dir.name}_{filename}"
            gallery.write_bytes(dest.read_bytes())
            written += 1

    sample_dirs = [
        (experiment_dir / "samples_custom", f"{title}: Custom samples", "samples_custom.png"),
        (experiment_dir / "samples_torch", f"{title}: Torch samples", "samples_torch.png"),
    ]
    for directory, sample_title, filename in sample_dirs:
        dest = plot_dir / filename
        plot_sample_grid(directory, sample_title, dest)
        if dest.is_file():
            gallery = gallery_dir / f"{experiment_dir.name}_{filename}"
            gallery.write_bytes(dest.read_bytes())
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DeepLearnLib training metrics CSVs.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Directory containing voc/, bccd/, synthetic/, cifar10/, and tabular/ result folders.",
    )
    return parser.parse_args()


def default_results_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent / "results"


def main() -> int:
    configure_style()
    args = parse_args()
    results_root = (args.results_root or default_results_root()).resolve()
    if not results_root.is_dir():
        print(f"[plot] Results directory does not exist: {results_root}", file=sys.stderr)
        print("[plot] Nothing to plot. Train a pipeline first, then re-run.")
        return 0

    gallery_dir = results_root / "plots"
    experiments = discover_experiments(results_root)
    if not experiments:
        print(f"[plot] No experiment folders with metrics CSVs under {results_root}")
        return 0

    total = 0
    for experiment_dir in experiments:
        total += process_experiment(experiment_dir, gallery_dir)

    print(f"[plot] Done. {total} figure(s) saved under {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
