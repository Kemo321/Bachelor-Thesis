#!/usr/bin/env python3
"""Generate thesis-quality PNG figures from pipeline metrics CSVs.

Looks for ``metrics_custom.csv`` and ``metrics_torch.csv`` under each experiment
directory (default: results/voc, results/bccd, results/synthetic). Missing Torch
files or mAP columns are skipped with a warning rather than aborting.
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

EXPERIMENT_DIRS = ("voc", "bccd", "synthetic")

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
        elif "train" in key:
            rename[column] = "TrainLoss"
        elif "test" in key or "val" in key:
            rename[column] = "TestLoss"
        elif "map" in key:
            rename[column] = "mAP@0.5"
        elif "time" in key:
            rename[column] = "Time(s)"
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


def experiment_title(name: str) -> str:
    mapping = {"voc": "PASCAL VOC", "bccd": "BCCD", "synthetic": "Synthetic"}
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
        ("map50.png", plot_map),
        ("epoch_duration.png", plot_epoch_duration),
    ]
    for filename, plotter in outputs:
        dest = plot_dir / filename
        plotter(custom, torch_df, title, dest)
        if dest.is_file():
            gallery = gallery_dir / f"{experiment_dir.name}_{filename}"
            gallery.parent.mkdir(parents=True, exist_ok=True)
            gallery.write_bytes(dest.read_bytes())
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DeepLearnLib training metrics CSVs.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Directory containing voc/, bccd/, and synthetic/ result folders.",
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
