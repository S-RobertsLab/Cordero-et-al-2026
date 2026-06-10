#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.signal import savgol_filter


# ============================= I/O helpers ============================= #

def read_fai(fai_path: Path) -> Dict[str, int]:
    """Read a .fai index → {chrom: length}."""
    chrom_lengths: Dict[str, int] = {}
    with fai_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            chrom, length = parts[0], int(parts[1])
            chrom_lengths[chrom] = length
    return chrom_lengths


def iter_bed_rows(bed_path: Path):
    """Yield (chrom, start, end, sample) from a simple BED-like file.
    Assumes SAMPLE is in the **last** column; ignores blank/comment lines."""
    with bed_path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                # Guard against malformed lines
                continue
            chrom = parts[0]
            start = int(parts[1])
            end   = int(parts[2])
            sample = parts[-1]
            yield chrom, start, end, sample


# ============================= Core logic ============================= #

def distance_to_nearest_telomere_bp(chrom_len: int, start: int, end: int) -> int:
    """Distance (bp) from an interval to the nearest chromosome end (telomere)."""
    # Distance to left telomere = start - 0 = start
    # Distance to right telomere = chrom_len - end
    left = start
    right = chrom_len - end
    return left if left <= right else right


def collect_distances_by_genotype(
    fai: Dict[str, int],
    bed_path: Path,
    genotype_length: int,
    skip_regex: str = r"(?:^chr?M$|^MT$)"
) -> Dict[str, Dict[str, Dict[int, int]]]:
    """
    Return:
      distances[genotype][sample][distance_bp] = count
    """
    distances: Dict[str, Dict[str, Dict[int, int]]] = {}
    rx_skip = re.compile(skip_regex, re.IGNORECASE)

    for chrom, start, end, sample in iter_bed_rows(bed_path):
        if chrom not in fai or rx_skip.search(chrom):
            continue

        # Genotype name from first N underscore-delimited tokens in SAMPLE
        toks = sample.split("_")
        genotype = "_".join(toks[:genotype_length]) if genotype_length > 0 else toks[0]

        d = distance_to_nearest_telomere_bp(fai[chrom], start, end)
        if d < 0:
            continue

        gdict = distances.setdefault(genotype, {})
        sdict = gdict.setdefault(sample, {})
        sdict[d] = sdict.get(d, 0) + 1

    return distances


def bin_by_distance(
    distances: Dict[str, Dict[str, Dict[int, int]]],
    bin_bp: int,
    max_bp: int | None
) -> Dict[str, Dict[str, Dict[int, int]]]:
    """Histogram each sample’s distance counts into fixed-width bins.
       Returns: binned[genotype][sample][bin_start_bp] = count"""
    binned: Dict[str, Dict[str, Dict[int, int]]] = {}

    for genotype, samples in distances.items():
        binned[genotype] = {}
        for sample, dd in samples.items():
            if not dd:
                binned[genotype][sample] = {}
                continue

            keys = np.fromiter(dd.keys(), dtype=int)
            vals = np.fromiter(dd.values(), dtype=int)

            dmax = int(keys.max())
            if max_bp is not None:
                dmax = min(dmax, int(max_bp))

            # ensure at least one bin
            edges = np.arange(0, dmax + bin_bp, bin_bp, dtype=int)
            if len(edges) < 2:
                edges = np.array([0, bin_bp], dtype=int)

            hist, edges = np.histogram(keys, bins=edges, weights=vals)
            # Map left edge → count
            binned[genotype][sample] = {int(edges[i]): int(hist[i]) for i in range(len(hist))}

    return binned


def combine_and_normalize(
    binned: Dict[str, Dict[str, Dict[int, int]]],
    bin_bp: int,
    normalize_by_samples: bool = True,
    normalize_to_per_mb: bool = True
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, int]]:
    """
    Combine samples within each genotype; optionally average across samples and
    convert to mutations per megabase (per-bin density).

    Returns:
      combined[genotype][bin_start_bp] = value
      nsamples[genotype] = number of samples in that genotype
    """
    combined: Dict[str, Dict[int, float]] = {}
    ns: Dict[str, int] = {}

    for genotype, samples in binned.items():
        acc: Dict[int, float] = {}
        for sample, series in samples.items():
            for bin_left, count in series.items():
                acc[bin_left] = acc.get(bin_left, 0.0) + float(count)

        n = max(1, len(samples))
        ns[genotype] = n

        if normalize_by_samples:
            for k in acc:
                acc[k] /= n

        if normalize_to_per_mb:
            # counts per bin → counts / bin_bp * 1e6 (mutations per Mb)
            scale = 1e6 / float(bin_bp)
            for k in acc:
                acc[k] *= scale

        # Ensure bins are dense from 0..max with step bin_bp (fill missing as 0.0)
        if acc:
            keys = sorted(acc.keys())
            kmax = keys[-1]
            dense = {i: acc.get(i, 0.0) for i in range(0, kmax + bin_bp, bin_bp)}
        else:
            dense = {}

        combined[genotype] = dense

    return combined, ns


def safe_savgol_with_zero_anchor(xs_bp: np.ndarray,
                                 ys: np.ndarray,
                                 window: int,
                                 poly: int,
                                 force_zero_at_origin: bool = True,
                                 zero_ramp_bins: int = 3) -> np.ndarray:
    """
    Edge-robust smoothing. Optionally anchors y(0)=0 and linearly ramps
    the first few bins up to the smoothed curve to avoid 'skyrocket' at 0.
    """
    n = len(ys)
    if n == 0:
        return ys
    if n < 3:
        out = ys.astype(float)
        if force_zero_at_origin and n >= 1:
            out[0] = 0.0
        return out

    # Ensure odd window and valid size
    if window % 2 == 0:
        window += 1
    window = min(window, n - (1 - n % 2))
    if window < poly + 2:
        window = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
        window = min(window, n - (1 - n % 2))
    if window < 3:
        out = ys.astype(float)
        if force_zero_at_origin and n >= 1:
            out[0] = 0.0
        return out

    # Smooth with a gentle edge mode
    try:
        sm = savgol_filter(ys, window_length=window, polyorder=min(poly, window - 1), mode="nearest")
    except Exception:
        sm = ys.astype(float)

    if force_zero_at_origin:
        out = sm.copy()
        # Force the first bin to 0 and ramp up smoothly over a few bins
        out[0] = 0.0
        k = max(1, min(zero_ramp_bins, n - 1))
        # Linear ramp from 0 at bin 0 to the smoothed value at bin k
        if k >= 1:
            yk = sm[k]
            ramp = np.linspace(0.0, yk, num=k + 1, endpoint=True)
            out[:k + 1] = ramp
        # Never go negative due to any numerical quirks
        out = np.maximum(out, 0.0)
        return out
    else:
        return np.maximum(sm, 0.0)



# ============================= Plotting ============================= #

def set_matplotlib_style():
    plt.rcParams.update({
        "figure.figsize": (5.0, 5.0),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.transparent": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })


def plot_per_genotype_svg(
    combined: Dict[str, Dict[int, float]],
    nsamples: Dict[str, int],
    outdir: Path,
    max_bp: int,
    show_scatter: bool,
    smoothing_window_pts: int,
    smoothing_poly: int,
    y_max: float | None
):
    outdir.mkdir(parents=True, exist_ok=True)
    set_matplotlib_style()

    for genotype, series in combined.items():
        if not series:
            continue

        xs_bp = np.array(sorted(series.keys()), dtype=int)
        ys = np.array([series[x] for x in xs_bp], dtype=float)

        # Limit to max_bp
        mask = xs_bp <= max_bp
        xs_bp = xs_bp[mask]
        ys = ys[mask]

        # Smooth
        ys_smooth = safe_savgol_with_zero_anchor(
            xs_bp, ys,
            window=smoothing_window_pts,
            poly=smoothing_poly,
            force_zero_at_origin=True,    # set False if you want the true peak at 0
            zero_ramp_bins=3              # increase if you want a longer gentle rise
        )


        fig, ax = plt.subplots()

        if show_scatter:
            ax.scatter(xs_bp / 1e3, ys, s=8, alpha=0.5, color="gray", label="Raw")

        ax.plot(xs_bp / 1e3, ys_smooth, color="black", label="Smoothed")

        ax.set_xlabel("Distance to Telomere (kb)")
        ax.set_ylabel("Mean Mutations per Bin (50bp)")
        ax.set_title(f"Mutations vs. Telomere Distance ({genotype})")

        # Legend with n
        ax.legend(title=f"n = {nsamples.get(genotype, 1)}", loc="upper right")

        # X ticks every 10 kb by default
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        # Y range
        if y_max is not None:
            ax.set_ylim(0, y_max)
        else:
            y_upper = np.nanmax(ys_smooth) if ys_smooth.size else 1.0
            ax.set_ylim(0, y_upper * 1.10 if y_upper > 0 else 1.0)

        # X range in kb
        ax.set_xlim(0, max_bp / 1e3)

        fig.tight_layout()
        out_path = outdir / f"{genotype}_telomere_distance.svg"
        fig.savefig(out_path, format="svg")
        plt.close(fig)


# ============================= CLI / Main ============================= #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot mutations vs. distance to nearest telomere (per genotype)."
    )
    p.add_argument("--bed", required=True, type=Path, help="BED-like file (chrom, start, end, ..., SAMPLE_last_col)")
    p.add_argument("--fai", required=True, type=Path, help="FASTA .fai index for chromosome lengths")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory for SVGs")
    p.add_argument("--genotype-length", type=int, default=2, help="Number of underscore-delimited tokens from SAMPLE to form genotype (default: 2).")
    p.add_argument("--bin-kb", type=float, default=0.05, help="Bin size in kilobases (default: 10 kb).")
    p.add_argument("--max-kb", type=float, default=100.0, help="Plot distances up to this many kilobases (default: 100 kb).")
    p.add_argument("--skip-chrom-regex", type=str, default=r"(?:^chr?M$|^MT$)", help="Regex of chrom names to skip (default skips mitochondrial).")
    p.add_argument("--no-average", action="store_true", help="Do not average across samples within a genotype.")
    p.add_argument("--no-per-mb", action="store_true", help="Do not normalize to per-megabase density.")
    p.add_argument("--scatter", action="store_true", help="Overlay raw per-bin scatter.")
    p.add_argument("--smooth-window", type=int, default=101, help="Savitzky-Golay smoothing window (points; adaptive if too large).")
    p.add_argument("--smooth-poly", type=int, default=3, help="Savitzky-Golay polynomial order.")
    p.add_argument("--ymax", type=float, default=None, help="Optional fixed Y-axis maximum.")
    return p.parse_args()


def main():
    args = parse_args()
    fai = read_fai(args.fai)

    distances = collect_distances_by_genotype(
        fai=fai,
        bed_path=args.bed,
        genotype_length=args.genotype_length,
        skip_regex=args.skip_chrom_regex
    )

    bin_bp = int(round(args.bin_kb * 1e3))
    max_bp = int(round(args.max_kb * 1e3))

    binned = bin_by_distance(distances, bin_bp=bin_bp, max_bp=max_bp)

    combined, nsamples = combine_and_normalize(
        binned=binned,
        bin_bp=bin_bp,
        normalize_by_samples=(not args.no_average),
        normalize_to_per_mb=(not args.no_per_mb),
    )

    plot_per_genotype_svg(
        combined=combined,
        nsamples=nsamples,
        outdir=args.outdir,
        max_bp=max_bp,
        show_scatter=args.scatter,
        smoothing_window_pts=args.smooth_window,
        smoothing_poly=args.smooth_poly,
        y_max=args.ymax
    )


if __name__ == "__main__":
    main()
