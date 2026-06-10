#!/usr/bin/env python3
"""
mutation_group_chi2.py
======================
χ² comparison of pooled mutation groups between two mutation-context matrices.

What’s new
----------
• By default we **combine exact reverse-complement channel pairs first** (e.g.,
  A[T>C]A + T[A>G]T), then run grouping and chi² on those combined values.
• Correct RC rule: reverse flanks and complement both flanks and mutation.
  Example: C[T>A]T → A[A>T]G (✓).
• Double-counting across different groups is allowed (by design).
  Within a single group, a pair contributes at most once.

Regions & Families (unchanged)
------------------------------
--subtelomere (default): keep strand-specific labels in groups
--internal              : labels include canonical pyrimidine groups;
                          we still keep purine-centric atypicals where defined.

--family both|atypical|canonical (default: both)

Complement handling
-------------------
--combine-complements / --no-complement-combine
  Default = --combine-complements:
     Build a pair-summed view: each exact channel and its exact RC are summed.
     Grouping uses this combined view. No canonicalization is applied.

  If you pass --no-complement-combine:
     Grouping uses the raw channels. You may also use --canonicalize then;
     canonicalization is IGNORED when complements are combined.

Groups (from Figure 5)
----------------------
Atypical (8):
  A[T>C]N, A[T>A]N, G[T>G]N, G[T>A]N, G[T>C]N, A[C>A]N, N[A>T]T, T[A>T]N
Canonical (5, pyrimidine-oriented):
  C[C>T]N, T[C>T]N, T[T>C]N, C[T>C]N, T[C>A]N

Output columns
--------------
mutation_type | mean_1 | mean_2 | log2FC | fold_change |
chi2_statistic | p_raw | p_fdr | signif
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# =============================================================================
# Reverse-complement & canonicalization helpers
# =============================================================================

_RC_BASE = {"A": "T", "T": "A", "C": "G", "G": "C"}
_STRICT_CH = re.compile(
    r"""^
        (?P<f>[ACGT])          # first flank
        \[
        (?P<ref>[ACGT])>(?P<alt>[ACGT])  # ref>alt
        \]
        (?P<l>[ACGT])          # last flank
        $""",
    re.VERBOSE
)

def rc_channel_exact(ch: str) -> str | None:
    """
    Reverse-complement an exact, fully-specified channel like 'C[T>A]T'.
    Returns 'A[A>T]G'. Returns None if not an exact channel.
    Rule: reverse flanks and complement both flanks and mutation.
          e.g., C[T>A]T  ->  A[A>T]G
    """
    m = _STRICT_CH.match(ch)
    if not m:
        return None
    f, l = m.group("f"), m.group("l")
    ref, alt = m.group("ref"), m.group("alt")
    f_rc = _RC_BASE[l]
    l_rc = _RC_BASE[f]
    ref_rc = _RC_BASE[ref]
    alt_rc = _RC_BASE[alt]
    return f"{f_rc}[{ref_rc}>{alt_rc}]{l_rc}"


# Canonicalization to pyrimidine-centered (kept for compat when NOT combining)
_COMP = str.maketrans("ATCG", "TAGC")
def canonical_channel(channel: str) -> str:
    """Reverse-complement purine-centred channel to pyrimidine form."""
    try:
        first = channel[0]
        ref, alt = channel[channel.find("[")+1:channel.find("]")].split(">")
        last = channel[-1]
        if ref in "AG":  # purine → RC to pyrimidine
            ctx = (first + ref + last).translate(_COMP)[::-1]
            new_ref = "T" if ref == "A" else "C"
            new_alt = {"T":"A","C":"G","G":"C","A":"T"}[alt]
            return f"{ctx[0]}[{new_ref}>{new_alt}]{ctx[2]}"
        return channel
    except Exception:
        return channel

def collapse_complements(df: pd.DataFrame) -> pd.DataFrame:
    mapping: OrderedDict[str, list[str]] = OrderedDict()
    for c in df.columns:
        mapping.setdefault(canonical_channel(c), []).append(c)
    return pd.DataFrame({k: df[v].mean(axis=1) for k, v in mapping.items()},
                        index=df.index)


# =============================================================================
# Figure-driven grouping rules
# =============================================================================

GROUPS_ATYPICAL = OrderedDict({
    "A[T>C]N": [r"^A\[T>C\][ACGT]$"],
    "A[T>A]N": [r"^A\[T>A\][ACGT]$"],
    "G[T>G]N": [r"^G\[T>G\][ACGT]$"],
    "G[T>A]N": [r"^G\[T>A\][ACGT]$"],
    "G[T>C]N": [r"^G\[T>C\][ACGT]$"],
    "A[C>A]N": [r"^A\[C>A\][ACGT]$"],
    "N[A>T]T": [r"^[ACGT]\[A>T\]T$"],
    "T[A>T]N": [r"^T\[A>T\][ACGT]$"],
})
GROUPS_CANONICAL = OrderedDict({
    "C[C>T]N": [r"^C\[C>T\][ACGT]$"],
    "T[C>T]N": [r"^T\[C>T\][ACGT]$"],
    "T[T>C]N": [r"^T\[T>C\][ACGT]$"],
    "C[T>C]N": [r"^C\[T>C\][ACGT]$"],
    "T[C>A]N": [r"^T\[C>A\][ACGT]$"],
})

def build_patterns(region: str, family: str) -> OrderedDict[str, list[str]]:
    """
    Build label -> regex list for the selected region & family.
    Prefix labels to disambiguate outputs.
    """
    out: OrderedDict[str, list[str]] = OrderedDict()
    def add(prefix: str, groups: OrderedDict[str, list[str]]):
        for k, v in groups.items():
            out[f"{prefix}{k}"] = v

    if region == "internal":
        if family in ("both", "atypical"):
            add("INT_ATYP::", GROUPS_ATYPICAL)
        if family in ("both", "canonical"):
            add("INT_CAN::", GROUPS_CANONICAL)
    else:
        if family in ("both", "atypical"):
            add("SUB_ATYP::", GROUPS_ATYPICAL)
        if family in ("both", "canonical"):
            add("SUB_CAN::", GROUPS_CANONICAL)
    return out


# =============================================================================
# Building a "combined complements" view (pair-summed channels)
# =============================================================================

def build_combined_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame where each exact channel and its exact RC have been **summed**
    into a single pair contribution, but exposed under both member names so that any
    regex (A-centric or its RC) can find the same combined value.

    Implementation detail:
      - Find all exact channels present.
      - For each channel 'ch', compute rc = rc_channel_exact(ch).
      - Pair-key = tuple(sorted([ch, rc])) (rc may be missing → pair is just ch).
      - Sum across the two Series that exist in df (missing treated as 0).
      - Materialize **both columns** (ch and rc) in the returned DF with the same
        combined Series so grouping works regardless of which side the regex targets.
      - Within a single group we deduplicate by pair-key to avoid double-adding.
    """
    exact_cols = [c for c in df.columns if _STRICT_CH.match(c)]
    # map: pair_key -> combined series
    pair_to_series: dict[tuple[str, str], pd.Series] = {}

    # Prepare zero series template
    zero = pd.Series(0.0, index=df.index)

    for ch in exact_cols:
        rc = rc_channel_exact(ch)
        if rc is None:
            continue
        # construct key with lexicographic order to make the pair canonical
        a, b = sorted([ch, rc])
        key = (a, b)
        if key in pair_to_series:
            continue
        s_a = df[a] if a in df.columns else zero
        s_b = df[b] if b in df.columns else zero
        pair_to_series[key] = s_a.add(s_b, fill_value=0.0)

    # Channels that are not exact (or unmatched) are carried over as-is
    out = pd.DataFrame(index=df.index)

    # Materialize combined values under BOTH member names
    for (a, b), s in pair_to_series.items():
        out[a] = s
        out[b] = s

    # Include any columns that are not part of an exact pair (e.g., if df has other measures)
    for c in df.columns:
        if _STRICT_CH.match(c):
            # it's exact; covered above (both a and b emitted)
            continue
        out[c] = df[c]

    # Ensure column order stable: keep original order when possible
    ordered_cols = [c for c in df.columns if c in out.columns] + [c for c in out.columns if c not in df.columns]
    return out[ordered_cols]


# =============================================================================
# Pooling
# =============================================================================

def pool_groups(df: pd.DataFrame,
                patterns: OrderedDict[str, list[str]],
                combined_mode: bool) -> pd.DataFrame:
    """
    Row-wise sums per selected group.

    When combined_mode=True, df is expected to be the **combined view** where
    each exact channel's column already holds the pair-summed values. To avoid
    double-adding within a group (e.g., both ATA and TAT match the same group's
    regex), we deduplicate by pair-key (computed on the column name) inside this
    function.
    """
    pooled: dict[str, pd.Series] = {}

    # Helper to compute pair key for an exact channel name
    def pair_key_for_col(col: str) -> tuple[str, str] | None:
        if not _STRICT_CH.match(col):
            return None
        rc = rc_channel_exact(col)
        if rc is None:
            return None
        return tuple(sorted([col, rc]))

    for group, regexes in patterns.items():
        pats = [re.compile(p) for p in regexes]
        cols = [c for c in df.columns if any(p.fullmatch(c) for p in pats)]
        if not cols:
            pooled[group] = pd.Series(0.0, index=df.index)
            continue

        # If we are in combined mode, deduplicate exact pairs within the group
        seen_pairs: set[tuple[str, str]] = set()
        contribs: list[pd.Series] = []
        for c in cols:
            if combined_mode:
                pk = pair_key_for_col(c)
                if pk is not None:
                    if pk in seen_pairs:
                        continue
                    seen_pairs.add(pk)
            contribs.append(df[c])

        pooled[group] = pd.concat(contribs, axis=1).sum(axis=1) if contribs else pd.Series(0.0, index=df.index)

    return pd.DataFrame(pooled, index=df.index)


# =============================================================================
# Stats (unchanged)
# =============================================================================

def bh_fdr(pvals: pd.Series) -> pd.Series:
    """Benjamini–Hochberg FDR (NaN-tolerant)."""
    q = pvals.copy()
    m = q.notna()
    if m.sum() == 0:
        return q
    p = q[m].values
    n = p.size
    order = np.argsort(p)[::-1]          # largest → smallest
    cummin = 1.0
    adj = np.empty_like(p)
    for i, idx in enumerate(order, start=1):
        cummin = min(cummin, p[idx] * n / (n - i + 1))
        adj[idx] = cummin
    q.loc[m] = np.clip(adj, 0, 1)
    return q

def stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

def run_chi2(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df1.columns if c in df2.columns]
    if not cols:
        raise ValueError("No shared mutation groups.")

    n1, n2 = len(df1), len(df2)
    eps = 1e-12
    rows = []
    for col in cols:
        mut1, mut2 = df1[col].sum(), df2[col].sum()
        mean1, mean2 = df1[col].mean(), df2[col].mean()
        fc = (mean2 + eps) / (mean1 + eps)
        log2fc = np.log2(fc)
        table = np.array([[mut1, n1], [mut2, n2]], float)
        try:
            chi2, p_raw, _, _ = chi2_contingency(table, correction=False)
        except ValueError:
            chi2, p_raw = np.nan, np.nan
        rows.append((col, mean1, mean2, log2fc, fc, chi2, p_raw))

    out = pd.DataFrame(rows,
                       columns=["mutation_type", "mean_1", "mean_2",
                                "log2FC", "fold_change",
                                "chi2_statistic", "p_raw"])
    out["p_fdr"] = bh_fdr(out["p_raw"])
    out["signif"] = out["p_fdr"].apply(stars)
    return out


# =============================================================================
# I/O
# =============================================================================

def read_clean(path: str | Path, delim: str, index_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=delim, engine="python")
    if index_col in df.columns:
        df.set_index(index_col, inplace=True)
    df = df.drop(columns=[c for c in ("Genotype_Treatment", "Total")
                          if c in df.columns], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    return df


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="χ² comparison of pooled mutation groups with optional complement-pair combining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("file1", help="First TSV/CSV file")
    ap.add_argument("file2", help="Second TSV/CSV file")
    ap.add_argument("-d", "--delimiter", default="\t", help="Field delimiter")
    ap.add_argument("-i", "--index_col", default="Unnamed: 0",
                    help="Column to use as replicate/sample index")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--subtelomere", dest="mode", action="store_const",
                      const="subtelomere",
                      help="Use Subtelomere groups (default).")
    mode.add_argument("--internal", dest="mode", action="store_const",
                      const="internal",
                      help="Use Internal groups.")
    ap.set_defaults(mode="subtelomere")

    ap.add_argument("--family", choices=["both", "atypical", "canonical"],
                    default="both",
                    help="Which family of groups to test within the chosen region")

    # Complement combining switch (default ON)
    cc = ap.add_mutually_exclusive_group()
    cc.add_argument("--combine-complements", dest="combine_complements",
                    action="store_true",
                    help="Sum exact reverse-complement channel pairs FIRST (default).")
    cc.add_argument("--no-complement-combine", dest="combine_complements",
                    action="store_false",
                    help="Do not pre-combine complements; use channels as-is.")
    ap.set_defaults(combine_complements=True)

    # Canonicalize kept for compatibility; ignored when combining complements
    ap.add_argument("-c", "--canonicalize", action="store_true",
                    help="Collapse complements to pyrimidine **only when** not combining complements.")

    ap.add_argument("-o", "--output", help="Output CSV path")
    return ap.parse_args(argv)


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    args = parse_args(argv)

    # Load raw numeric matrices
    try:
        df1_raw = read_clean(args.file1, args.delimiter, args.index_col)
        df2_raw = read_clean(args.file2, args.delimiter, args.index_col)
    except Exception as e:
        sys.exit(f"[error] failed to load inputs: {e}")

    # Build the working view for each file
    if args.combine_complements:
        # Pair-summed view (preferred): A[T>X]A + T[A>Y]T etc.
        df1 = build_combined_view(df1_raw)
        df2 = build_combined_view(df2_raw)
        canonical_flag = False  # prevent double-handling
    else:
        # Raw view; allow optional canonicalization for internal-style analyses
        df1 = collapse_complements(df1_raw) if args.canonicalize else df1_raw
        df2 = collapse_complements(df2_raw) if args.canonicalize else df2_raw

    # Build patterns for region+family
    patterns = build_patterns(args.mode, args.family)

    # Pool per selected groups (dedup within-group by exact RC pair if combined)
    try:
        df1p = pool_groups(df1, patterns, combined_mode=args.combine_complements)
        df2p = pool_groups(df2, patterns, combined_mode=args.combine_complements)
    except Exception as e:
        sys.exit(f"[error] pooling failed: {e}")

    # χ² tests
    res = run_chi2(df1p, df2p)
    if res.empty:
        sys.exit("[warning] nothing to test after pooling.")

    # Output
    suffix = []
    suffix.append(args.mode)
    suffix.append(args.family)
    suffix.append("COMB" if args.combine_complements else ("CANON" if args.canonicalize else "RAW"))

    out_path = (Path(args.output)
                if args.output
                else Path(args.file1).with_name(
                    f"{Path(args.file1).stem}_vs_{Path(args.file2).stem}_{'_'.join(suffix)}_pooled_chi2.csv"))
    res.to_csv(out_path, index=False)
    print(f"✓ {len(res)} tests completed → {out_path}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
