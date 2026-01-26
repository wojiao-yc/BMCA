#!/usr/bin/env python
import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _mean_sem(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x, np.zeros_like(x), 1
    n = x.shape[0]
    mean = np.nanmean(x, axis=0)
    if n <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = np.nanstd(x, axis=0) / np.sqrt(n)
    return mean, sem, n


def _default_out_png(npz_path: str) -> str:
    base, ext = os.path.splitext(str(npz_path))
    if ext.lower() == ".npz":
        return base + ".png"
    return npz_path + ".png"


def _blur_radii_from_name(npz_path: Path) -> Optional[Tuple[str, str]]:
    match = re.search(r"(?:orig|blur)(\d+)_vs(\d+)", npz_path.stem, re.IGNORECASE)
    if not match:
        return None
    left = str(int(match.group(1)))
    right = str(int(match.group(2)))
    return left, right


def plot_from_npz(npz_path: str, out_png: str, dpi: int = 200) -> None:
    data = np.load(str(npz_path))

    if "times_centers" in data:
        times = data["times_centers"]
    elif "times" in data:
        times = data["times"]
    else:
        raise KeyError("Missing key 'times_centers' (or 'times') in npz.")

    if "rho_orig" not in data or "rho_blur" not in data:
        raise KeyError("Missing key 'rho_orig' or 'rho_blur' in npz.")

    rho_orig = data["rho_orig"]
    rho_blur = data["rho_blur"]

    mean_orig, sem_orig, n_sub = _mean_sem(rho_orig)
    mean_blur, sem_blur, _ = _mean_sem(rho_blur)

    blur_radii = _blur_radii_from_name(Path(npz_path))
    if blur_radii:
        radius_left, radius_right = blur_radii
        label_orig = f"EEG vs Blur radius {radius_left}"
        label_blur = f"EEG vs Blur radius {radius_right}"
        title = "Raw-EEG RSA (mean ± SEM)"
        partial_label_orig = (
            f"EEG vs Blur radius {radius_left} | Blur radius {radius_right}"
        )
        partial_label_blur = (
            f"EEG vs Blur radius {radius_right} | Blur radius {radius_left}"
        )
        partial_title = "Partial RSA (mean ± SEM)"
    else:
        label_orig = "RSA: EEG RDM vs Orig Img RDM"
        label_blur = "RSA: EEG RDM vs Blur Img RDM"
        title = "Raw-EEG RSA (mean ± SEM)"
        partial_label_orig = "Partial RSA: EEG vs Orig | Blur"
        partial_label_blur = "Partial RSA: EEG vs Blur | Orig"
        partial_title = "Partial RSA (mean ± SEM)"

    plt.figure(figsize=(10, 4))
    plt.plot(times, mean_orig, label=label_orig)
    plt.fill_between(times, mean_orig - sem_orig, mean_orig + sem_orig, alpha=0.2)
    plt.plot(times, mean_blur, label=label_blur)
    plt.fill_between(times, mean_blur - sem_blur, mean_blur + sem_blur, alpha=0.2)
    plt.xlabel("Time (s, window center)")
    plt.ylabel("Spearman corr")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

    has_partial = ("prho_blur_given_orig" in data) and ("prho_orig_given_blur" in data)
    if not has_partial:
        return

    prho_blur = data["prho_blur_given_orig"]
    prho_orig = data["prho_orig_given_blur"]

    mean_p_blur, sem_p_blur, _ = _mean_sem(prho_blur)
    mean_p_orig, sem_p_orig, _ = _mean_sem(prho_orig)

    stem, ext = os.path.splitext(out_png)
    if not ext:
        ext = ".png"
    out_partial = f"{stem}_partial{ext}"

    plt.figure(figsize=(10, 4))
    plt.plot(times, mean_p_orig, label=partial_label_orig)
    plt.fill_between(times, mean_p_orig - sem_p_orig, mean_p_orig + sem_p_orig, alpha=0.2)
    plt.plot(times, mean_p_blur, label=partial_label_blur)
    plt.fill_between(times, mean_p_blur - sem_p_blur, mean_p_blur + sem_p_blur, alpha=0.2)
    plt.xlabel("Time (s, window center)")
    plt.ylabel("Partial Spearman corr")
    plt.title(partial_title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_partial, dpi=dpi)
    plt.close()


def _iter_npz_files(in_dir: Path, glob_pattern: str, recursive: bool):
    if recursive:
        yield from in_dir.rglob(glob_pattern)
    else:
        yield from in_dir.glob(glob_pattern)


def _out_png_for_file(npz_path: Path, out_dir: Optional[Path], out_png: Optional[str]) -> Path:
    if out_png:
        return Path(out_png)
    if out_dir:
        return out_dir / f"{npz_path.stem}.png"
    return Path(_default_out_png(npz_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz", help="Input npz path (from erp_rdm_rsa.py)")
    group.add_argument("--in_dir", help="Input directory containing npz files")
    parser.add_argument("--glob", default="*.npz", help="Glob pattern used with --in_dir")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories of --in_dir")
    parser.add_argument("--out_dir", default=None, help="Output directory for plots")
    parser.add_argument("--out_png", default=None, help="Output png path (single file only)")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if args.in_dir:
        if args.out_png:
            raise ValueError("--out_png is only valid with --npz; use --out_dir for batches.")
        in_dir = Path(args.in_dir)
        if not in_dir.is_dir():
            raise FileNotFoundError(f"--in_dir not found: {in_dir}")
        out_dir = Path(args.out_dir) if args.out_dir else in_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for npz_path in _iter_npz_files(in_dir, args.glob, args.recursive):
            if not npz_path.is_file():
                continue
            out_png = out_dir / f"{npz_path.stem}.png"
            plot_from_npz(npz_path, out_png, dpi=args.dpi)
            print(f"Saved plot: {out_png}")
        return

    npz_path = Path(args.npz)
    if not npz_path.is_file():
        raise FileNotFoundError(f"--npz not found: {npz_path}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    out_png = _out_png_for_file(npz_path, out_dir, args.out_png)
    plot_from_npz(npz_path, out_png, dpi=args.dpi)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
