#!/usr/bin/env python3

import argparse
import os
import math
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Force a non-interactive backend before importing any Matplotlib plotting layer.
# This prevents X11/Qt/Tk pixmap allocation failures on headless systems.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Sample/genotype map
# -----------------------------------------------------------------------------
# This script keeps only BAMs whose filename contains one of these sample-code
# identifiers, then uses the mapped genotype/condition label for the output
# tables and figure grouping.
#
# Important: these codes are intentionally treated as full alphanumeric sample
# identifiers. For example, B285, C285, A285, and 285B are distinct conditions.
SAMPLE_CODE_TO_GENOTYPE = {
    "285B": "WT 150J",
    "286B": "WT 150J",
    "J378": "POLE NXXXN 150J",
    "J379": "POLE NXXXN 150J",
    "D014": "PR 6-4PL 450J",
    "D002": "PR 6-4PL 450J",
    "01B": "MPH1 KO 150J",
    "03B": "RAD16 KO 150J",
    "C285": "PR WT 450J",
    "C286": "PR WT 450J",
    "B285": "WT 450J",
    "B286": "WT 450J",
    "G002": "RAD16 KO 150J",
    "G014": "RAD16 KO 150J",
    "A285": "WT 150J",
    "A286": "WT 150J",
    "416B": "POLH KO 150J",
    "285n": "WT NT",
    "01n": "MPH1 NT",
    "03n": "RAD16 NT",
    "416n": "POLH NT",
}

GENOTYPE_DISPLAY_ORDER = [
    "WT 150J",
    "POLH KO 150J",
    "POLE NXXXN 150J",
    "MPH1 KO 150J",
    "RAD16 KO 150J",
    "WT 450J",
    "PR WT 450J",
    "PR 6-4PL 450J",
    "WT NT",
    "MPH1 NT",
    "RAD16 NT",
    "POLH NT"
]

SAMPLE_CODE_LOOKUP = {
    code.upper(): (code, genotype)
    for code, genotype in SAMPLE_CODE_TO_GENOTYPE.items()
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Call telomere-loss events from BAM depth. The script tiles the last "
            "N bp of each chromosome arm into fixed-size chunks, computes each "
            "chunk's mean depth, normalizes each chunk to the sample's internal "
            "mean depth, and calls a telomere lost if any chunk on that arm is "
            "below the chosen depth-ratio threshold."
        )
    )
    parser.add_argument(
        "--bam-dir",
        help="Directory containing BAM files. Required unless --depth-table is provided.",
    )
    parser.add_argument(
        "--fai",
        help="Reference .fai file. Required unless --depth-table is provided.",
    )
    parser.add_argument(
        "--depth-table",
        help=(
            "Existing *.chunk_depth.tsv table from a previous run. If provided, "
            "BAMs are not opened and samtools is not run. Use this to re-call "
            "loss events with a different --loss-threshold."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output prefix, or an old-style output table path. If a suffix is "
            "given, its stem is used as the prefix. Example: results/tel_loss "
            "creates results/tel_loss.chunk_depth.tsv, results/tel_loss.telomere_calls.tsv, etc."
        ),
    )
    parser.add_argument(
        "--tel-size",
        type=int,
        default=20000,
        help="Telomeric/subtelomeric span from each chromosomal end in bp. Default: 20000",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size used to tile each telomere arm in bp. Default: 1000",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=0.10,
        help=(
            "Call a chunk lost when chunk_mean_depth / internal_mean_depth is "
            "strictly below this value. Default: 0.10"
        ),
    )
    parser.add_argument(
        "--samtools",
        default="samtools",
        help="Path to samtools executable. Default: samtools",
    )
    parser.add_argument(
        "--include-mito",
        action="store_true",
        help="Include mitochondrial contigs. By default likely mitochondrial contigs are excluded.",
    )
    parser.add_argument(
        "--index-missing",
        action="store_true",
        help="Run samtools index for BAMs that lack an index.",
    )
    parser.add_argument(
        "--index-threads",
        type=int,
        default=11,
        help="Threads for samtools index -@. Default: 11",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of BAMs to process in parallel. Use 1 for serial execution. Default: 1",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write the chunk/telomere/sample TSV outputs but skip figure generation. Useful on headless systems or very large sample sets.",
    )
    parser.add_argument(
        "--max-figure-width",
        type=float,
        default=24.0,
        help="Maximum figure width in inches for the SVG. Default: 24.0",
    )
    parser.add_argument(
        "--min-internal-depth",
        type=float,
        default=None,
        help=(
            "Exclude samples from genotype summary/plot when their internal mean "
            "sequencing depth is below this value. Raw chunk/telomere/sample TSVs "
            "are still written with QC flags. Example: --min-internal-depth 20"
        ),
    )
    parser.add_argument(
        "--internal-depth-sd-filter",
        type=float,
        default=None,
        help=(
            "Exclude samples from genotype summary/plot when internal mean depth is "
            "more than this many standard deviations from the cohort mean. Use 2 "
            "for a ±2 SD filter. Default: no SD filter."
        ),
    )
    parser.add_argument(
        "--internal-depth-sd-scope",
        choices=["all", "genotype"],
        default="all",
        help=(
            "Scope used for --internal-depth-sd-filter. 'all' computes one cohort "
            "mean/SD across all samples; 'genotype' computes mean/SD within each "
            "genotype. Default: all."
        ),
    )
    return parser.parse_args()


def output_prefix_from_arg(output_arg: str) -> Path:
    path = Path(output_arg)
    if path.suffix:
        return path.with_suffix("")
    return path


def prefixed_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"


def natural_key(value):
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(x)) if x.isdigit() else (1, x.lower()) for x in parts)


def genotype_sort_key(value):
    value = str(value)
    if value in GENOTYPE_DISPLAY_ORDER:
        return (0, GENOTYPE_DISPLAY_ORDER.index(value), natural_key(value))
    return (1, len(GENOTYPE_DISPLAY_ORDER), natural_key(value))


def parse_sample_identity_from_name(path: Path):
    stem = path.stem

    tokens = [t for t in re.split(r"[_\-.+\s]+", stem) if t]

    for token in tokens:
        token_upper = token.upper()
        if token_upper in SAMPLE_CODE_LOOKUP:
            canonical_code, genotype = SAMPLE_CODE_LOOKUP[token_upper]
            prefix_match = re.match(r"^[A-Za-z]+", canonical_code)
            return {
                "sample_id_token": token,
                "sample_prefix": prefix_match.group(0) if prefix_match else "",
                "sample_code": canonical_code,
                "genotype": genotype,
            }

    stem_upper = stem.upper()
    for code_upper, (canonical_code, genotype) in sorted(
        SAMPLE_CODE_LOOKUP.items(), key=lambda item: len(item[0]), reverse=True
    ):
        pattern = rf"(?<![A-Z0-9]){re.escape(code_upper)}(?![A-Z0-9])"
        match = re.search(pattern, stem_upper)
        if match:
            matched_token = stem[match.start():match.end()]
            prefix_match = re.match(r"^[A-Za-z]+", canonical_code)
            return {
                "sample_id_token": matched_token,
                "sample_prefix": prefix_match.group(0) if prefix_match else "",
                "sample_code": canonical_code,
                "genotype": genotype,
            }

    return None


def is_probably_mito(contig_name: str) -> bool:
    name = contig_name.strip().lower()
    return (
        name.startswith("mit")
        or name in {"mt", "chrm", "chrmt", "mtdna", "mitochondrion_genome"}
    )


def load_fai(fai_path: Path, include_mito: bool) -> pd.DataFrame:
    fai = pd.read_csv(
        fai_path,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["contig", "length"],
    )
    fai["length"] = fai["length"].astype(int)

    if not include_mito:
        fai = fai[~fai["contig"].map(is_probably_mito)].copy()

    if fai.empty:
        raise ValueError("No contigs remained after parsing/filtering the FAI file.")

    return fai


def build_internal_regions_from_fai(fai_df: pd.DataFrame, tel_size: int) -> pd.DataFrame:
    rows = []
    for contig, length in fai_df.itertuples(index=False):
        length = int(length)
        start = tel_size
        end = length - tel_size
        if end > start:
            rows.append(
                {
                    "contig": contig,
                    "start": start,
                    "end": end,
                    "region_id": f"{contig}:internal",
                }
            )

    internal_df = pd.DataFrame(rows)

    if internal_df.empty:
        raise ValueError(
            "No internal regions were defined. This usually means every contig is "
            "shorter than 2 * --tel-size."
        )

    internal_df["bp"] = internal_df["end"] - internal_df["start"]
    return internal_df


def build_telomere_chunks_from_fai(
    fai_df: pd.DataFrame,
    tel_size: int,
    chunk_size: int,
) -> pd.DataFrame:
    if tel_size <= 0:
        raise ValueError("--tel-size must be > 0.")
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0.")
    if chunk_size > tel_size:
        raise ValueError("--chunk-size cannot be larger than --tel-size.")

    rows = []

    for contig, length in fai_df.itertuples(index=False):
        length = int(length)
        if length <= 0:
            continue

        # Left arm: chunk index 1 is closest to the left chromosome end.
        left_window_end = min(tel_size, length)
        chunk_index = 1
        for start in range(0, left_window_end, chunk_size):
            end = min(start + chunk_size, left_window_end)
            if end <= start:
                continue
            distance_from_end = start
            rows.append(
                {
                    "contig": contig,
                    "contig_length": length,
                    "telomere_id": f"{contig}_L",
                    "telomere_side": "left",
                    "chunk_index_from_end": chunk_index,
                    "distance_from_end_bp": distance_from_end,
                    "region_start_0based": start,
                    "region_end_0based": end,
                    "chunk_bp": end - start,
                    "region_id": (
                        f"{contig}_L_chunk{chunk_index:03d}_"
                        f"{start}_{end}"
                    ),
                }
            )
            chunk_index += 1

        # Right arm: chunk index 1 is closest to the right chromosome end.
        right_window_start = max(0, length - tel_size)
        chunk_index = 1
        end = length
        while end > right_window_start:
            start = max(right_window_start, end - chunk_size)
            distance_from_end = length - end
            rows.append(
                {
                    "contig": contig,
                    "contig_length": length,
                    "telomere_id": f"{contig}_R",
                    "telomere_side": "right",
                    "chunk_index_from_end": chunk_index,
                    "distance_from_end_bp": distance_from_end,
                    "region_start_0based": start,
                    "region_end_0based": end,
                    "chunk_bp": end - start,
                    "region_id": (
                        f"{contig}_R_chunk{chunk_index:03d}_"
                        f"{start}_{end}"
                    ),
                }
            )
            end = start
            chunk_index += 1

    tel_df = pd.DataFrame(rows)

    if tel_df.empty:
        raise ValueError("No telomere chunks were defined from the provided FAI file.")

    tel_df = tel_df.sort_values(
        ["contig", "telomere_side", "chunk_index_from_end"],
        key=lambda s: s.map(natural_key) if s.name == "contig" else s,
        kind="mergesort",
    ).reset_index(drop=True)

    return tel_df


def write_bed_from_df(df: pd.DataFrame, out_path: Path, include_region_metadata: bool):
    with open(out_path, "w") as handle:
        for row in df.itertuples(index=False):
            if include_region_metadata:
                handle.write(
                    "\t".join(
                        [
                            str(row.contig),
                            str(row.region_start_0based),
                            str(row.region_end_0based),
                            str(row.region_id),
                            str(row.telomere_id),
                            str(row.telomere_side),
                            str(row.chunk_index_from_end),
                            str(row.distance_from_end_bp),
                        ]
                    )
                    + "\n"
                )
            else:
                handle.write(f"{row.contig}\t{row.start}\t{row.end}\n")


def bam_has_index(bam_path: Path) -> bool:
    possible_indexes = [
        Path(str(bam_path) + ".bai"),
        bam_path.with_suffix(".bai"),
        Path(str(bam_path) + ".csi"),
        bam_path.with_suffix(".csi"),
    ]
    return any(p.exists() for p in possible_indexes)


def ensure_bam_index(
    bam_path: Path,
    samtools: str,
    index_missing: bool,
    index_threads: int,
):
    if bam_has_index(bam_path):
        return

    if not index_missing:
        raise FileNotFoundError(
            f"No BAM index found for {bam_path}. Create one first or rerun with --index-missing."
        )

    cmd = [samtools, "index", "-@", str(index_threads), str(bam_path)]
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"samtools index failed for {bam_path}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{exc.stderr}"
        ) from exc


def run_bedcov_sum(bed_path: Path, bam_path: Path, samtools: str) -> float:
    cmd = [samtools, "bedcov", str(bed_path), str(bam_path)]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"samtools bedcov failed for {bam_path}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{exc.stderr}"
        ) from exc

    total_depth_sum = 0.0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        total_depth_sum += float(fields[-1])

    return total_depth_sum


def run_bedcov_telomere_chunks(
    bed_path: Path,
    bam_path: Path,
    samtools: str,
) -> pd.DataFrame:
    cmd = [samtools, "bedcov", str(bed_path), str(bam_path)]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"samtools bedcov failed for {bam_path}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{exc.stderr}"
        ) from exc

    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9:
            raise ValueError(
                "Unexpected samtools bedcov output. Expected the original BED8 "
                "columns plus a final depth-sum column."
            )

        contig = fields[0]
        start = int(fields[1])
        end = int(fields[2])
        region_id = fields[3]
        telomere_id = fields[4]
        telomere_side = fields[5]
        chunk_index_from_end = int(fields[6])
        distance_from_end_bp = int(fields[7])
        depth_sum = float(fields[-1])

        rows.append(
            {
                "contig": contig,
                "region_start_0based": start,
                "region_end_0based": end,
                "region_id": region_id,
                "telomere_id": telomere_id,
                "telomere_side": telomere_side,
                "chunk_index_from_end": chunk_index_from_end,
                "distance_from_end_bp": distance_from_end_bp,
                "depth_sum": depth_sum,
            }
        )

    return pd.DataFrame(rows)


def process_one_bam(
    bam_path: Path,
    telomere_chunks_bed: Path,
    internal_bed: Path,
    telomere_chunks_df: pd.DataFrame,
    samtools: str,
    internal_bp: int,
    index_missing: bool,
    index_threads: int,
):
    ensure_bam_index(bam_path, samtools, index_missing, index_threads)

    identity = parse_sample_identity_from_name(bam_path)
    if identity is None:
        raise ValueError(
            f"Could not map BAM filename to one of the requested sample codes: {bam_path.name}"
        )

    sample = bam_path.stem
    internal_sum = run_bedcov_sum(internal_bed, bam_path, samtools)
    internal_mean = internal_sum / internal_bp if internal_bp > 0 else np.nan

    bedcov_df = run_bedcov_telomere_chunks(telomere_chunks_bed, bam_path, samtools)

    # Merge back to the FAI-derived metadata to carry contig length and chunk_bp.
    chunk_df = bedcov_df.merge(
        telomere_chunks_df[
            [
                "region_id",
                "contig_length",
                "chunk_bp",
            ]
        ],
        on="region_id",
        how="left",
        validate="one_to_one",
    )

    if chunk_df["chunk_bp"].isna().any():
        missing = chunk_df.loc[chunk_df["chunk_bp"].isna(), "region_id"].head(5).tolist()
        raise ValueError(f"Could not recover chunk metadata for regions: {missing}")

    chunk_df["chunk_bp"] = chunk_df["chunk_bp"].astype(int)
    chunk_df["chunk_mean_depth"] = chunk_df["depth_sum"] / chunk_df["chunk_bp"]

    if np.isfinite(internal_mean) and internal_mean > 0:
        chunk_df["chunk_depth_ratio_to_internal"] = (
            chunk_df["chunk_mean_depth"] / internal_mean
        )
    else:
        chunk_df["chunk_depth_ratio_to_internal"] = np.nan

    for key, value in {
        "sample": sample,
        "sample_id_token": identity["sample_id_token"],
        "sample_prefix": identity["sample_prefix"],
        "sample_code": identity["sample_code"],
        "genotype": identity["genotype"],
        "bam_path": str(bam_path),
        "internal_bp": internal_bp,
        "internal_depth_sum": internal_sum,
        "internal_mean_depth": internal_mean,
    }.items():
        chunk_df[key] = value

    ordered_cols = [
        "sample",
        "sample_id_token",
        "sample_prefix",
        "sample_code",
        "genotype",
        "bam_path",
        "contig",
        "contig_length",
        "telomere_id",
        "telomere_side",
        "chunk_index_from_end",
        "distance_from_end_bp",
        "region_start_0based",
        "region_end_0based",
        "chunk_bp",
        "depth_sum",
        "chunk_mean_depth",
        "internal_bp",
        "internal_depth_sum",
        "internal_mean_depth",
        "chunk_depth_ratio_to_internal",
        "region_id",
    ]

    return chunk_df[ordered_cols]


def collect_depths_parallel(
    bam_files,
    telomere_chunks_bed: Path,
    internal_bed: Path,
    telomere_chunks_df: pd.DataFrame,
    samtools: str,
    internal_bp: int,
    index_missing: bool,
    index_threads: int,
    workers: int,
) -> pd.DataFrame:
    dfs = []

    if workers <= 1:
        for i, bam in enumerate(bam_files, start=1):
            print(f"[{i}/{len(bam_files)}] Processing {bam.name}", file=sys.stderr)
            dfs.append(
                process_one_bam(
                    bam_path=bam,
                    telomere_chunks_bed=telomere_chunks_bed,
                    internal_bed=internal_bed,
                    telomere_chunks_df=telomere_chunks_df,
                    samtools=samtools,
                    internal_bp=internal_bp,
                    index_missing=index_missing,
                    index_threads=index_threads,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_bam = {
                executor.submit(
                    process_one_bam,
                    bam,
                    telomere_chunks_bed,
                    internal_bed,
                    telomere_chunks_df,
                    samtools,
                    internal_bp,
                    index_missing,
                    index_threads,
                ): bam
                for bam in bam_files
            }

            done_count = 0
            for future in as_completed(future_to_bam):
                bam = future_to_bam[future]
                dfs.append(future.result())
                done_count += 1
                print(f"[{done_count}/{len(bam_files)}] Finished {bam.name}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    return sort_depth_table(df)


def sort_depth_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["_genotype_sort"] = df["genotype"].map(genotype_sort_key)
    df["_sample_code_sort"] = df["sample_code"].map(natural_key)
    df["_sample_sort"] = df["sample"].map(natural_key)
    df["_contig_sort"] = df["contig"].map(natural_key)
    df["_side_sort"] = df["telomere_side"].map({"left": 0, "right": 1}).fillna(9)

    df = (
        df.sort_values(
            [
                "_genotype_sort",
                "_sample_code_sort",
                "_sample_sort",
                "_contig_sort",
                "_side_sort",
                "chunk_index_from_end",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_genotype_sort",
                "_sample_code_sort",
                "_sample_sort",
                "_contig_sort",
                "_side_sort",
            ]
        )
        .reset_index(drop=True)
    )

    return df


def validate_depth_table(df: pd.DataFrame):
    required = {
        "sample",
        "sample_code",
        "genotype",
        "contig",
        "telomere_id",
        "telomere_side",
        "chunk_index_from_end",
        "distance_from_end_bp",
        "region_start_0based",
        "region_end_0based",
        "chunk_bp",
        "depth_sum",
        "chunk_mean_depth",
        "internal_mean_depth",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "The --depth-table is missing required columns: " + ", ".join(missing)
        )


def load_depth_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    validate_depth_table(df)

    numeric_cols = [
        "chunk_index_from_end",
        "distance_from_end_bp",
        "region_start_0based",
        "region_end_0based",
        "chunk_bp",
        "depth_sum",
        "chunk_mean_depth",
        "internal_mean_depth",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "chunk_depth_ratio_to_internal" not in df.columns:
        df["chunk_depth_ratio_to_internal"] = np.where(
            df["internal_mean_depth"] > 0,
            df["chunk_mean_depth"] / df["internal_mean_depth"],
            np.nan,
        )
    else:
        df["chunk_depth_ratio_to_internal"] = pd.to_numeric(
            df["chunk_depth_ratio_to_internal"],
            errors="coerce",
        )

    return sort_depth_table(df)


def add_chunk_loss_calls(df: pd.DataFrame, loss_threshold: float) -> pd.DataFrame:
    if not (0 <= loss_threshold <= 1):
        raise ValueError("--loss-threshold should be between 0 and 1.")

    called = df.copy()

    if "chunk_depth_ratio_to_internal" not in called.columns:
        called["chunk_depth_ratio_to_internal"] = np.where(
            called["internal_mean_depth"] > 0,
            called["chunk_mean_depth"] / called["internal_mean_depth"],
            np.nan,
        )

    called["loss_threshold"] = float(loss_threshold)
    called["chunk_callable"] = np.isfinite(called["chunk_depth_ratio_to_internal"])
    called["chunk_lost"] = (
        called["chunk_callable"]
        & (called["chunk_depth_ratio_to_internal"] < loss_threshold)
    )

    return called


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def summarize_telomere_calls(chunk_calls: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouping_cols = [
        "sample",
        "sample_id_token",
        "sample_prefix",
        "sample_code",
        "genotype",
        "bam_path",
        "contig",
        "contig_length",
        "telomere_id",
        "telomere_side",
        "internal_mean_depth",
        "loss_threshold",
    ]
    grouping_cols = [col for col in grouping_cols if col in chunk_calls.columns]

    for keys, g in chunk_calls.groupby(grouping_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping_cols, keys))

        callable_g = g[g["chunk_callable"]].copy()
        lost_g = callable_g[callable_g["chunk_lost"]].copy()

        telomere_callable = len(callable_g) > 0
        telomere_lost = bool(lost_g.shape[0] > 0) if telomere_callable else np.nan

        row.update(
            {
                "n_chunks": int(g.shape[0]),
                "n_callable_chunks": int(callable_g.shape[0]),
                "n_lost_chunks": int(lost_g.shape[0]),
                "telomere_callable": telomere_callable,
                "telomere_lost": telomere_lost,
                "min_chunk_depth_ratio_to_internal": (
                    float(callable_g["chunk_depth_ratio_to_internal"].min())
                    if telomere_callable
                    else np.nan
                ),
                "mean_chunk_depth_ratio_to_internal": weighted_mean(
                    callable_g["chunk_depth_ratio_to_internal"],
                    callable_g["chunk_bp"],
                ),
                "mean_telomere_depth": weighted_mean(
                    callable_g["chunk_mean_depth"],
                    callable_g["chunk_bp"],
                ),
                "first_lost_chunk_index_from_end": (
                    int(lost_g["chunk_index_from_end"].min())
                    if not lost_g.empty
                    else np.nan
                ),
                "first_lost_chunk_distance_from_end_bp": (
                    int(lost_g["distance_from_end_bp"].min())
                    if not lost_g.empty
                    else np.nan
                ),
                "lost_chunk_indices_from_end": (
                    ",".join(map(str, sorted(lost_g["chunk_index_from_end"].astype(int).unique())))
                    if not lost_g.empty
                    else ""
                ),
            }
        )
        rows.append(row)

    tel_df = pd.DataFrame(rows)

    if tel_df.empty:
        return tel_df

    tel_df["_genotype_sort"] = tel_df["genotype"].map(genotype_sort_key)
    tel_df["_sample_code_sort"] = tel_df["sample_code"].map(natural_key)
    tel_df["_sample_sort"] = tel_df["sample"].map(natural_key)
    tel_df["_contig_sort"] = tel_df["contig"].map(natural_key)
    tel_df["_side_sort"] = tel_df["telomere_side"].map({"left": 0, "right": 1}).fillna(9)

    tel_df = (
        tel_df.sort_values(
            [
                "_genotype_sort",
                "_sample_code_sort",
                "_sample_sort",
                "_contig_sort",
                "_side_sort",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_genotype_sort",
                "_sample_code_sort",
                "_sample_sort",
                "_contig_sort",
                "_side_sort",
            ]
        )
        .reset_index(drop=True)
    )

    return tel_df


def summarize_sample_losses(telomere_calls: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouping_cols = [
        "sample",
        "sample_id_token",
        "sample_prefix",
        "sample_code",
        "genotype",
        "bam_path",
        "internal_mean_depth",
        "loss_threshold",
    ]
    grouping_cols = [col for col in grouping_cols if col in telomere_calls.columns]

    for keys, g in telomere_calls.groupby(grouping_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping_cols, keys))

        callable_g = g[g["telomere_callable"] == True].copy()
        lost_g = callable_g[callable_g["telomere_lost"] == True].copy()

        telomeres_evaluated = int(g.shape[0])
        telomeres_callable = int(callable_g.shape[0])
        telomeres_lost = int(lost_g.shape[0])

        row.update(
            {
                "telomeres_evaluated": telomeres_evaluated,
                "telomeres_callable": telomeres_callable,
                "telomeres_lost": telomeres_lost,
                "fraction_telomeres_lost": (
                    telomeres_lost / telomeres_callable
                    if telomeres_callable > 0
                    else np.nan
                ),
                "lost_telomere_ids": (
                    ",".join(lost_g["telomere_id"].astype(str).tolist())
                    if not lost_g.empty
                    else ""
                ),
            }
        )
        rows.append(row)

    sample_df = pd.DataFrame(rows)
    if sample_df.empty:
        return sample_df

    sample_df["_genotype_sort"] = sample_df["genotype"].map(genotype_sort_key)
    sample_df["_sample_code_sort"] = sample_df["sample_code"].map(natural_key)
    sample_df["_sample_sort"] = sample_df["sample"].map(natural_key)

    sample_df = (
        sample_df.sort_values(
            ["_genotype_sort", "_sample_code_sort", "_sample_sort"],
            kind="mergesort",
        )
        .drop(columns=["_genotype_sort", "_sample_code_sort", "_sample_sort"])
        .reset_index(drop=True)
    )

    return sample_df


def apply_internal_depth_qc(
    sample_summary: pd.DataFrame,
    min_internal_depth: float | None = None,
    internal_depth_sd_filter: float | None = None,
    sd_scope: str = "all",
) -> pd.DataFrame:
    """
    Add sample-level internal-depth QC flags.

    The raw telomere-loss calls are not changed. These flags are used to decide
    which samples contribute to the genotype-level summary and figure.
    """
    df = sample_summary.copy()

    if df.empty:
        df["internal_depth_qc_pass"] = pd.Series(dtype=bool)
        df["internal_depth_qc_reason"] = pd.Series(dtype=str)
        return df

    if "internal_mean_depth" not in df.columns:
        raise ValueError(
            "sample_summary is missing internal_mean_depth; cannot apply internal-depth QC."
        )

    df["internal_mean_depth"] = pd.to_numeric(df["internal_mean_depth"], errors="coerce")
    df["internal_depth_qc_min_threshold"] = (
        float(min_internal_depth) if min_internal_depth is not None else np.nan
    )
    df["internal_depth_qc_sd_filter"] = (
        float(internal_depth_sd_filter) if internal_depth_sd_filter is not None else np.nan
    )
    df["internal_depth_qc_sd_scope"] = sd_scope

    df["internal_depth_qc_mean"] = np.nan
    df["internal_depth_qc_sd"] = np.nan
    df["internal_depth_zscore"] = np.nan

    pass_mask = np.isfinite(df["internal_mean_depth"].to_numpy(dtype=float))
    reasons = [[] for _ in range(len(df))]

    # Minimum absolute internal-depth filter, e.g. exclude <20x.
    if min_internal_depth is not None:
        min_internal_depth = float(min_internal_depth)
        below_min = df["internal_mean_depth"] < min_internal_depth
        pass_mask &= ~below_min.to_numpy(dtype=bool)
        for i, flag in enumerate(below_min.to_numpy(dtype=bool)):
            if flag:
                reasons[i].append(f"internal_depth_below_{min_internal_depth:g}x")

    # Mean ± N SD internal-depth filter. This removes both under- and over-sequenced samples.
    if internal_depth_sd_filter is not None:
        internal_depth_sd_filter = float(internal_depth_sd_filter)
        if internal_depth_sd_filter <= 0:
            raise ValueError("--internal-depth-sd-filter must be > 0 when provided.")
        if sd_scope not in {"all", "genotype"}:
            raise ValueError("--internal-depth-sd-scope must be 'all' or 'genotype'.")

        if sd_scope == "all":
            groups = [(None, df.index)]
        else:
            groups = list(df.groupby("genotype", dropna=False, sort=False).groups.items())

        for _, idx in groups:
            idx = pd.Index(idx)
            vals = df.loc[idx, "internal_mean_depth"].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.sum() < 2:
                # With fewer than 2 finite samples, SD is undefined; do not exclude by SD.
                group_mean = float(np.nanmean(vals)) if finite.sum() else np.nan
                group_sd = np.nan
                z = np.full(len(vals), np.nan)
                outlier = np.zeros(len(vals), dtype=bool)
            else:
                group_mean = float(np.nanmean(vals))
                group_sd = float(np.nanstd(vals, ddof=1))
                if group_sd == 0 or not np.isfinite(group_sd):
                    z = np.full(len(vals), 0.0)
                    outlier = np.zeros(len(vals), dtype=bool)
                else:
                    z = (vals - group_mean) / group_sd
                    outlier = np.isfinite(z) & (np.abs(z) > internal_depth_sd_filter)

            df.loc[idx, "internal_depth_qc_mean"] = group_mean
            df.loc[idx, "internal_depth_qc_sd"] = group_sd
            df.loc[idx, "internal_depth_zscore"] = z

            idx_list = list(idx)
            for local_i, flag in enumerate(outlier):
                if flag:
                    global_i = df.index.get_loc(idx_list[local_i])
                    reasons[global_i].append(
                        f"internal_depth_gt_{internal_depth_sd_filter:g}sd_from_{sd_scope}_mean"
                    )
            pass_mask[df.index.get_indexer(idx)] &= ~outlier

    df["internal_depth_qc_pass"] = pass_mask
    df["plot_included"] = pass_mask
    df["internal_depth_qc_reason"] = [";".join(r) if r else "pass" for r in reasons]

    return df


def summarize_genotype_losses(sample_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize sample-level telomere-loss counts by genotype.

    Each sample contributes one value: telomeres_lost. This is the exact data
    plotted as individual points in the genotype-level figure.
    """
    rows = []
    if sample_summary.empty:
        return pd.DataFrame()

    summary_for_stats = sample_summary.copy()
    if "internal_depth_qc_pass" in summary_for_stats.columns:
        summary_for_stats = summary_for_stats[summary_for_stats["internal_depth_qc_pass"] == True].copy()

    if summary_for_stats.empty:
        return pd.DataFrame(
            columns=[
                "genotype",
                "n_samples",
                "mean_telomeres_lost_per_sample",
                "median_telomeres_lost_per_sample",
                "sd_telomeres_lost_per_sample",
                "sem_telomeres_lost_per_sample",
                "min_telomeres_lost",
                "max_telomeres_lost",
                "total_telomeres_lost",
            ]
        )

    for genotype in sorted(summary_for_stats["genotype"].dropna().unique(), key=genotype_sort_key):
        g = summary_for_stats[summary_for_stats["genotype"] == genotype].copy()
        values = pd.to_numeric(g["telomeres_lost"], errors="coerce").dropna().to_numpy(dtype=float)
        n = int(len(values))
        rows.append(
            {
                "genotype": genotype,
                "n_samples": n,
                "mean_telomeres_lost_per_sample": float(np.mean(values)) if n else np.nan,
                "median_telomeres_lost_per_sample": float(np.median(values)) if n else np.nan,
                "sd_telomeres_lost_per_sample": float(np.std(values, ddof=1)) if n >= 2 else 0.0,
                "sem_telomeres_lost_per_sample": sem(values) if n else np.nan,
                "min_telomeres_lost": float(np.min(values)) if n else np.nan,
                "max_telomeres_lost": float(np.max(values)) if n else np.nan,
                "total_telomeres_lost": int(np.sum(values)) if n else 0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_genotype_sort"] = out["genotype"].map(genotype_sort_key)
    out = out.sort_values("_genotype_sort", kind="mergesort").drop(columns=["_genotype_sort"]).reset_index(drop=True)
    return out


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def make_telomere_loss_figure(
    sample_summary: pd.DataFrame,
    genotype_summary: pd.DataFrame,
    out_svg: Path,
    tel_size: int,
    chunk_size: int,
    loss_threshold: float,
    max_figure_width: float = 14.0,
):
    """
    Plot only genotype-level telomere-loss burden.

    Each sample contributes one plotted point equal to the number of lost
    telomeres/chromosome arms in that sample. The white square is the genotype
    mean ± SEM. Figure width scales with genotype count, not sample count, so it
    should remain lightweight even for many samples.
    """
    if sample_summary.empty:
        raise ValueError("No sample-level loss calls found; cannot plot.")

    df = sample_summary.copy()
    if "internal_depth_qc_pass" in df.columns:
        df = df[df["internal_depth_qc_pass"] == True].copy()
    df["telomeres_lost"] = pd.to_numeric(df["telomeres_lost"], errors="coerce")
    df = df.dropna(subset=["genotype", "telomeres_lost"])
    if df.empty:
        raise ValueError(
            "No QC-passing samples with finite telomeres_lost values found; cannot plot. "
            "Relax --min-internal-depth or --internal-depth-sd-filter, or rerun with --no-plot."
        )

    genotypes = sorted(df["genotype"].unique(), key=genotype_sort_key)
    fig_width = min(max(7.0, 1.15 * len(genotypes) + 3.0), max_figure_width)
    fig = Figure(figsize=(fig_width, 5.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    prop_cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    genotype_colors = {g: prop_cycle[i % len(prop_cycle)] for i, g in enumerate(genotypes)}
    rng = np.random.default_rng(12345)
    group_positions = np.arange(len(genotypes), dtype=float)

    for i, genotype in enumerate(genotypes):
        g = df[df["genotype"] == genotype].copy()
        y = g["telomeres_lost"].to_numpy(dtype=float)
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(y))

        # One point per sample; this is the biological replicate level.
        ax.scatter(
            np.full(len(y), group_positions[i]) + jitter,
            y,
            s=60,
            alpha=0.9,
            color=genotype_colors[genotype],
            edgecolors="black",
            linewidths=0.35,
            zorder=2,
        )

        mean_y = float(np.mean(y)) if len(y) else np.nan
        sem_y = sem(y)
        ax.errorbar(
            [group_positions[i]],
            [mean_y],
            yerr=[sem_y],
            fmt="s",
            mfc="white",
            mec="black",
            ecolor="black",
            capsize=4,
            ms=7,
            zorder=3,
        )

        ax.text(
            group_positions[i],
            mean_y + sem_y + max(0.25, 0.03 * max(float(df["telomeres_lost"].max()), 1.0)),
            f"mean={mean_y:.2f}\nn={len(y)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    y_max = float(df["telomeres_lost"].max()) if len(df) else 1.0
    ax.set_xticks(group_positions)
    ax.set_xticklabels(genotypes, rotation=25, ha="right")
    ax.set_ylabel("Lost telomeres / chromosome arms per sample")
    ax.set_xlabel("Genotype")
    ax.set_title("Telomere-loss burden by genotype")
    ax.set_ylim(bottom=0, top=max(1.0, y_max + max(1.0, 0.18 * y_max)))
    ax.margins(x=0.08)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="black", label="Sample"),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            label="Mean ± SEM",
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")

    fig.suptitle(
        (
            f"Telomere loss: any {chunk_size:,} bp chunk within terminal {tel_size:,} bp "
            f"with depth <{loss_threshold:.3g}× sample internal mean"
        ),
        y=1.02,
        fontsize=11,
    )

    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.clear()


def save_tsv(df: pd.DataFrame, out_path: Path):
    df.to_csv(out_path, sep="\t", index=False)


def main():
    args = parse_args()

    output_prefix = output_prefix_from_arg(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    chunk_depth_path = prefixed_path(output_prefix, ".chunk_depth.tsv")
    chunk_calls_path = prefixed_path(output_prefix, ".chunk_calls.tsv")
    telomere_calls_path = prefixed_path(output_prefix, ".telomere_calls.tsv")
    sample_summary_path = prefixed_path(output_prefix, ".sample_loss_summary.tsv")
    genotype_summary_path = prefixed_path(output_prefix, ".genotype_loss_summary.tsv")
    figure_path = prefixed_path(output_prefix, ".telomere_loss.svg")

    if args.depth_table:
        depth_table_path = Path(args.depth_table)
        if not depth_table_path.exists():
            sys.exit(f"--depth-table does not exist: {depth_table_path}")

        print(f"Loading existing depth table: {depth_table_path}", file=sys.stderr)
        chunk_depth_df = load_depth_table(depth_table_path)

    else:
        if args.bam_dir is None or args.fai is None:
            sys.exit("--bam-dir and --fai are required unless --depth-table is provided.")

        if shutil.which(args.samtools) is None:
            sys.exit(f"Could not find samtools executable: {args.samtools}")

        bam_dir = Path(args.bam_dir)
        fai_path = Path(args.fai)

        if not bam_dir.exists():
            sys.exit(f"BAM directory does not exist: {bam_dir}")
        if not fai_path.exists():
            sys.exit(f"FAI file does not exist: {fai_path}")

        all_bam_files = sorted(bam_dir.glob("*.bam"), key=lambda p: natural_key(p.name))
        if len(all_bam_files) == 0:
            sys.exit(f"No BAM files found in: {bam_dir}")

        bam_files = [
            bam for bam in all_bam_files
            if parse_sample_identity_from_name(bam) is not None
        ]
        skipped_bam_files = [
            bam for bam in all_bam_files
            if parse_sample_identity_from_name(bam) is None
        ]

        if len(bam_files) == 0:
            wanted = ", ".join(SAMPLE_CODE_TO_GENOTYPE.keys())
            sys.exit(
                f"No BAM files matched the requested sample-code map in: {bam_dir}\n"
                f"Expected one of these sample codes in the filename: {wanted}"
            )

        print(
            f"Keeping {len(bam_files)} mapped BAM(s); skipping {len(skipped_bam_files)} unmapped BAM(s).",
            file=sys.stderr,
        )
        if skipped_bam_files:
            preview = ", ".join(p.name for p in skipped_bam_files[:10])
            suffix = " ..." if len(skipped_bam_files) > 10 else ""
            print(f"Skipped examples: {preview}{suffix}", file=sys.stderr)

        fai_df = load_fai(fai_path, include_mito=args.include_mito)
        telomere_chunks_df = build_telomere_chunks_from_fai(
            fai_df,
            tel_size=args.tel_size,
            chunk_size=args.chunk_size,
        )
        internal_df = build_internal_regions_from_fai(fai_df, tel_size=args.tel_size)
        internal_bp = int(internal_df["bp"].sum())

        print(
            (
                f"Defined {len(telomere_chunks_df)} telomere chunks "
                f"({args.tel_size:,} bp per arm, {args.chunk_size:,} bp chunks) "
                f"and {internal_bp:,} internal bp."
            ),
            file=sys.stderr,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            telomere_chunks_bed = tmpdir / "telomere_chunks.bed"
            internal_bed = tmpdir / "internal.bed"

            write_bed_from_df(
                telomere_chunks_df,
                telomere_chunks_bed,
                include_region_metadata=True,
            )
            write_bed_from_df(
                internal_df,
                internal_bed,
                include_region_metadata=False,
            )

            chunk_depth_df = collect_depths_parallel(
                bam_files=bam_files,
                telomere_chunks_bed=telomere_chunks_bed,
                internal_bed=internal_bed,
                telomere_chunks_df=telomere_chunks_df,
                samtools=args.samtools,
                internal_bp=internal_bp,
                index_missing=args.index_missing,
                index_threads=args.index_threads,
                workers=args.workers,
            )

        save_tsv(chunk_depth_df, chunk_depth_path)
        print(f"Reusable chunk-depth table: {chunk_depth_path}", file=sys.stderr)

    chunk_calls_df = add_chunk_loss_calls(chunk_depth_df, args.loss_threshold)
    telomere_calls_df = summarize_telomere_calls(chunk_calls_df)
    sample_summary_df = summarize_sample_losses(telomere_calls_df)
    sample_summary_df = apply_internal_depth_qc(
        sample_summary_df,
        min_internal_depth=args.min_internal_depth,
        internal_depth_sd_filter=args.internal_depth_sd_filter,
        sd_scope=args.internal_depth_sd_scope,
    )
    genotype_summary_df = summarize_genotype_losses(sample_summary_df)

    save_tsv(chunk_calls_df, chunk_calls_path)
    save_tsv(telomere_calls_df, telomere_calls_path)
    save_tsv(sample_summary_df, sample_summary_path)
    save_tsv(genotype_summary_df, genotype_summary_path)

    if "internal_depth_qc_pass" in sample_summary_df.columns:
        n_total = int(sample_summary_df.shape[0])
        n_pass = int((sample_summary_df["internal_depth_qc_pass"] == True).sum())
        n_fail = n_total - n_pass
        print(
            f"Internal-depth QC: {n_pass}/{n_total} samples included for genotype summary/plot; {n_fail} excluded.",
            file=sys.stderr,
        )

    if args.no_plot:
        print("Figure generation skipped because --no-plot was provided.", file=sys.stderr)
    else:
        make_telomere_loss_figure(
            sample_summary=sample_summary_df,
            genotype_summary=genotype_summary_df,
            out_svg=figure_path,
            tel_size=args.tel_size,
            chunk_size=args.chunk_size,
            loss_threshold=args.loss_threshold,
            max_figure_width=args.max_figure_width,
        )

    print(f"Chunk-level calls:     {chunk_calls_path}")
    print(f"Telomere-level calls:  {telomere_calls_path}")
    print(f"Sample-level summary:  {sample_summary_path}")
    print(f"Genotype summary:      {genotype_summary_path}")
    if not args.depth_table:
        print(f"Reusable depth table:  {chunk_depth_path}")
    if not args.no_plot:
        print(f"Figure:                {figure_path}")


if __name__ == "__main__":
    main()
