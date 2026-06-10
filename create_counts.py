#!/usr/bin/env python3
"""
Create 192-context SNV spectra and full DNV spectra, binned into
  right-telomere · internal  (cut-off = 20 kb by default).

Left-telomere rows are reverse-complemented and merged into the right row.
"""

from pathlib import Path
from collections import defaultdict
import itertools as it
import pickle, sys
import pandas as pd
from Bio import SeqIO
from Bio.Seq import reverse_complement as rc

# ─────────────────────────── key lists ────────────────────────────
REFS = "ACGT"
ALTS = {b: [x for x in REFS if x != b] for b in REFS}

SNV_KEYS = [f"{l}[{ref}>{alt}]{r}"
            for l, r, ref in it.product(REFS, REFS, REFS)
            for alt in ALTS[ref]]

DINUCS   = [''.join(p) for p in it.product(REFS, repeat=2)]
DNV_KEYS = [f"{x}>{y}" for x in DINUCS for y in DINUCS if x != y]

# ────────────────── reverse-complement helpers (robust) ──────────────────
def rc_key_snv(k: str) -> str:
    l, ref, alt, r = k[0], k[2], k[4], k[-1]
    return f"{rc(r)}[{rc(ref)}>{rc(alt)}]{rc(l)}"

def rc_key_dnv(k: str) -> str:
    """Reverse-complement a dinucleotide key of the form 'AC>GT'.

    If *k* is malformed (no '>'), return it unchanged so merge() won’t crash.
    """
    if ">" not in k:
        return k                     # leave the oddball column as-is
    ref, alt = k.split(">", 1)       # at most one split
    return f"{rc(ref)}>{rc(alt)}"

# ───────────────────────── region helper ──────────────────────────
def region_of(pos, length, cutoff):                    # pos is 0-based
    if pos < cutoff: return "left"
    if length - pos < cutoff: return "right"
    return "internal"

# ──────────────────── cache & denominator counting ────────────────
def cache_path(fasta, cutoff): return Path(fasta).with_suffix(f'.kmerCounts_cut{cutoff}.pkl')

def count_kmers_2_3_by_region(fasta, cutoff):
    """
    Count 3-mers (tri) and 2-mers (di) by genomic region in a single pass.

    Mitochondrial contigs are skipped – i.e. any record whose .id or .name
    contains the substring 'mit' or equals 'm' (case-insensitive).
    """
    tri = defaultdict(lambda: defaultdict(int))
    di  = defaultdict(lambda: defaultdict(int))
    refs = set("ACGT")

    for rec in SeqIO.parse(fasta, "fasta"):
        name_lc = rec.id.lower()
        if "mit" in name_lc or name_lc == "m":
            continue                                  # ← skip mitochondria

        seq = str(rec.seq).upper()
        L   = len(seq)

        for i in range(L - 2):
            k3 = seq[i : i + 3]
            if set(k3) <= refs:
                region = region_of(i + 1, L, cutoff)  # anchor = middle base
                tri[region][k3] += 1

            k2 = k3[:2]
            if set(k2) <= refs:
                region = region_of(i + 1, L, cutoff)  # anchor = 2nd base
                di[region][k2] += 1

        # last 2-mer at the end of the chromosome
        k2 = seq[-2:]
        if set(k2) <= refs:
            di[region_of(L - 1, L, cutoff)][k2] += 1

    # cast defaultdicts → plain dicts for clean pickling
    tri = {reg: dict(cnts) for reg, cnts in tri.items()}
    di  = {reg: dict(cnts) for reg, cnts in di.items()}
    return tri, di


def load_or_build_denoms(fasta, cutoff):
    """
    Load cached denominators if present; otherwise build them (skipping
    mitochondrial contigs as per count_kmers_2_3_by_region) and cache.
    """
    pkl = cache_path(fasta, cutoff)

    if pkl.exists():
        try:
            return pickle.load(pkl.open("rb"))
        except Exception:
            pkl.unlink(missing_ok=True)

    tri, di = count_kmers_2_3_by_region(fasta, cutoff)

    with pkl.open("wb") as fh:
        pickle.dump({"tri": tri, "di": di}, fh)

    return {"tri": tri, "di": di}

# ───────────────────────────── main ───────────────────────────────
def build_tables(bed, fai, cutoff=20_000):
    bed   = Path(bed)
    stem  = bed.with_suffix('')
    fasta = str(Path(fai).with_suffix(''))

    # chromosome lengths
    chr_len = {ln.split()[0]: int(ln.split()[1]) for ln in open(fai)}

    # denominators
    denoms = load_or_build_denoms(fasta, cutoff)
    tri, di = denoms['tri'], denoms['di']

    snv_cnt = defaultdict(lambda: defaultdict(int))
    dnv_cnt = defaultdict(lambda: defaultdict(int))
    refs = set(REFS); skipped = 0

    for ln in bed.open():
        f = ln.rstrip('\n').split('\t')
        if len(f) < 10: continue
        chrom, pos = f[0], int(f[1])
        ctx, mut, mtype, sample = f[6].upper(), f[7].upper(), f[-2], f[-1]
        if 'mit' in chrom.lower(): continue
        L = chr_len.get(chrom) or chr_len.get(chrom.split('_')[0])
        if L is None: skipped += 1; continue
        region = region_of(pos, L, cutoff)

        if mtype == 'SNV' and set(ctx) <= refs and mut[0] in REFS and mut[-1] in REFS:
            snv_cnt[(sample, region)][f"{ctx[0]}[{mut}]{ctx[-1]}"] += 1
        elif mtype == 'DNV':
            ref, alt = mut.split('>')
            if set(ref + alt) <= refs:
                dnv_cnt[(sample, region)][mut] += 1
    if skipped:
        print(f"[WARN] {skipped} rows skipped due to unmatched chromosomes.", file=sys.stderr)

    # tidy helper (pre-seed zeros)
    def tidy(data, keys):
        return (pd.DataFrame(data)
                  .T.reindex(columns=keys, fill_value=0)
                  .fillna(0).astype(int)
                  .rename_axis(['sample', 'region']))

    snv_raw = tidy(snv_cnt, SNV_KEYS)
    dnv_raw = tidy(dnv_cnt, DNV_KEYS)

    # ──── FIXED normalise() ────
    def normalise(raw, denom_tab, k):
        out = raw.astype(float).copy()
        for (sample, region), row in raw.iterrows():
            d = denom_tab[region]
            sites = ([c[0] + c[2] + c[-1] for c in row.index] if k == 3
                     else [c.split('>')[0] for c in row.index])
            denom = pd.Series([d.get(s, 0) for s in sites], index=row.index)
            denom.replace(0, pd.NA, inplace=True)
            out.loc[(sample, region)] = (row / denom * 1e6).fillna(0)
        return out

    snv_rate = normalise(snv_raw, tri, 3)
    dnv_rate = normalise(dnv_raw, di, 2)

    def merge(df, rc_fun):
        """
        Collapse the left- and right-telomere rows for each sample into a single
        'subtelomere' row whose values are the **mean** of the two (rather than
        the sum).  
        • If only one side is present, its values are used as-is.  
        • The left row is reverse-complement–mapped via *rc_fun* before averaging.
        """
        df   = df.copy()
        cols = df.columns

        for samp in df.index.get_level_values(0).unique():
            parts = []

            # right telomere (if any)
            if (samp, "right") in df.index:
                parts.append(df.loc[(samp, "right")])
                df = df.drop(index=(samp, "right"))

            # left telomere (if any) – reverse-complement the contexts first
            if (samp, "left") in df.index:
                parts.append(df.loc[(samp, "left")].rename(index=rc_fun))
                df = df.drop(index=(samp, "left"))

            # nothing to merge
            if not parts:
                continue

            # average (or single row if only one side)
            subtel = pd.concat(parts, axis=1).mean(axis=1)

            # explicit row assignment to avoid accidental column creation
            df.loc[(samp, "subtelomere"), :] = subtel.values

        return df

    snv_cnt_m  = merge(snv_raw,  rc_key_snv)
    snv_rate_m = merge(snv_rate, rc_key_snv)
    dnv_cnt_m  = merge(dnv_raw,  rc_key_dnv)
    dnv_rate_m = merge(dnv_rate, rc_key_dnv)

    out_dir, name = stem.parent, stem.name
    snv_cnt_m.to_csv(out_dir / f"{name}_SNV_192_counts.tsv", sep='\t')
    snv_rate_m.to_csv(out_dir / f"{name}_SNV_192_rates.tsv",  sep='\t')
    dnv_cnt_m.to_csv(out_dir / f"{name}_DNV_counts.tsv",      sep='\t')
    dnv_rate_m.to_csv(out_dir / f"{name}_DNV_rates.tsv",      sep='\t')

# ───────────────────────── example ─────────────────────────
if __name__ == "__main__":
    FAI = "path/to/fai"
    BED = "path/to/bed"
    build_tables(BED, FAI, cutoff=0)
