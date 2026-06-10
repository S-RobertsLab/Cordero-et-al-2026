import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as patches
import matplotlib as mpl
from pathlib import Path

# Disable LaTeX rendering (if desired)
mpl.rcParams['text.usetex'] = False  

def canonical_channel(channel):
    """
    Given a channel string in the format X[Y>Z]W, return its canonical
    pyrimidine–centric representation. If the ref base (Y) is a purine (A or G),
    convert by taking the reverse complement.
    
    For example:
      "T[G>T]T"  --> "A[C>A]A"
      "A[C>A]A"  stays as "A[C>A]A" because it is already pyrimidine–centric.
    """
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    try:
        # Split out the parts: first letter, the mutation inside, and the last letter.
        first = channel[0]
        inside = channel[channel.find('[')+1: channel.find(']')]
        ref, alt = inside.split('>')
        last = channel[-1]
        context = first + ref + last
        # If the reference is a purine, convert to pyrimidine–centric by taking the reverse complement.
        if ref in 'AG':
            # Compute the reverse complement of the 3–letter context.
            new_context = "".join(comp[b] for b in reversed(context))
            # Convert the ref and alt according to the rule:
            # (if ref=='A', then new ref becomes 'T'; if ref=='G', new ref becomes 'C'; etc.)
            new_ref = 'T' if ref == 'A' else 'C'
            # For alt, use the same mapping as in your original code:
            if alt == 'T':
                new_alt = 'A'
            elif alt == 'C':
                new_alt = 'G'
            elif alt == 'G':
                new_alt = 'C'
            elif alt == 'A':
                new_alt = 'T'
            else:
                new_alt = alt
            return f"{new_context[0]}[{new_ref}>{new_alt}]{new_context[2]}"
        else:
            # Already in canonical (pyrimidine) form
            return channel
    except Exception:
        return channel

def complement_channel(channel):
    """
    Compute the complementary channel label for a canonical channel.
    For example, for canonical "A[C>A]A", returns "T[G>T]T".
    """
    comp = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    try:
        first = channel[0]
        inside = channel[channel.find('[')+1: channel.find(']')]
        ref, alt = inside.split('>')
        last = channel[-1]
        # Reverse complement the 3-letter context:
        new_context = "".join(comp[b] for b in reversed(first+ref+last))
        # For the mutation itself, convert ref and alt:
        if ref == 'T':
            new_ref = 'A'
        elif ref == 'C':
            new_ref = 'G'
        elif ref == 'G':
            new_ref = 'C'
        elif ref == 'A':
            new_ref = 'T'
        else:
            new_ref = ref

        if alt == 'T':
            new_alt = 'A'
        elif alt == 'C':
            new_alt = 'G'
        elif alt == 'G':
            new_alt = 'C'
        elif alt == 'A':
            new_alt = 'T'
        else:
            new_alt = alt
        return f"{new_context[0]}[{new_ref}>{new_alt}]{new_context[2]}"
    except Exception:
        return channel

def get_complementary_color(rgb):
    """
    Given an RGB tuple (values between 0 and 1), return its complementary color.
    If the original color is black (very close to (1/256, 1/256, 1/256)), return medium gray instead.
    """
    # Define the original black color used in mutation_colors
    black_rgb = (1/256, 1/256, 1/256)
    
    # Compute complementary color
    comp_color = (1 - rgb[0], 1 - rgb[1], 1 - rgb[2])
    
    # If the original color is black, set the complement to gray manually
    if np.allclose(rgb, black_rgb, atol=0.01):
        return (0.5, 0.5, 0.5)  # Medium gray
    
    return comp_color


def plot_combined_96_channel_mutation_spectrum_from_tsv(file_path, show_error_bars=True):
    """
    Read a TSV file that has 192 mutation channel columns (i.e. separate counts
    for each strand), combine complementary channels into 96 canonical channels
    by taking the *mean* of the two strands (instead of the sum), and plot the
    spectrum (mean ± std dev across samples).
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path, sep="\t")
    
    # Remove summary rows: drop rows where the first column is empty or equals "Mean"
    df_samples = df[~df["Unnamed: 0"].astype(str).str.strip().isin(["", "Mean"])].copy()
    
    # Identify the mutation channel columns (all except known extras)
    excluded_cols = ["Unnamed: 0", "Genotype_Treatment", "Total"]
    channel_cols = [col for col in df_samples.columns if col not in excluded_cols]
    
    # Make sure these columns are numeric
    df_samples[channel_cols] = df_samples[channel_cols].apply(pd.to_numeric)
    
    # Map each column name (192 channels) to its canonical (pyrimidine-centric) channel
    mapping = {col: canonical_channel(col) for col in channel_cols}
    
    # Group columns by canonical channel
    combined_dict = {}
    for col, canon in mapping.items():
        combined_dict.setdefault(canon, []).append(col)
    
    # Build a new DataFrame with one column per canonical channel — *mean* of strands
    df_canonical = pd.DataFrame(index=df_samples.index)
    for canon, cols in combined_dict.items():
        df_canonical[canon] = df_samples[cols].mean(axis=1)   # ← changed from .sum(...)
    
    # Compute per-channel means and standard deviations across samples
    channel_means = df_canonical.mean()
    channel_stds  = df_canonical.std()
    
    # --- Define the standard order for the 96 channels ---
    bases = ['A', 'C', 'G', 'T']
    mutation_types = [f"{b1}[{ref}>{alt}]{b2}" 
                      for ref in 'CT' 
                      for alt in 'ACGT' if ref != alt 
                      for b1 in bases 
                      for b2 in bases]
    
    # Fill missing channels with zero counts
    for mtype in mutation_types:
        if mtype not in channel_means:
            channel_means[mtype] = 0
            channel_stds[mtype] = 0
    
    # Reorder statistics according to the standard order
    channel_means = channel_means.reindex(mutation_types)
    channel_stds  = channel_stds.reindex(mutation_types)
    
    # --- Define colors for each substitution type ---
    mutation_colors = {
        'C>A': [3/256, 189/256, 239/256],   # Light blue
        'C>G': [1/256, 1/256, 1/256],       # Black
        'C>T': [228/256, 41/256, 38/256],   # Red
        'T>A': [203/256, 202/256, 202/256], # Grey
        'T>C': [162/256, 207/256, 99/256],  # Green
        'T>G': [236/256, 199/256, 197/256]  # Salmon
    }
    colors = [mutation_colors[mut[2:5]] for mut in mutation_types]
    
    # --- Begin plotting ---
    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(14, 4))
    
    x = np.arange(len(mutation_types))
    ax.bar(x, channel_means.values,
           yerr=channel_stds.values if show_error_bars else None,
           color=colors, edgecolor='none', capsize=4, alpha=0.8,
           error_kw={'elinewidth':0.5, 'ecolor':'grey', 'capthick':0.5} if show_error_bars else {})
    
    ax.set_xlim(-0.5, len(mutation_types) - 0.5)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.set_xticks([])
    
    # Custom rotated labels below the bars
    for i, label in enumerate(mutation_types):
        first = label[0]
        mid   = label[2]
        last  = label[-1]
        r, g, b = colors[i]
        x_pos = x[i]
        y_pos = -0.02
        ax.text(x_pos, y_pos, last, ha='center', va='top', rotation=90,
                fontsize=10, fontweight='bold', color="dimgrey",
                fontfamily="DejaVu Sans Mono",
                transform=ax.get_xaxis_transform())
        ax.text(x_pos, y_pos - 0.03, mid, ha='center', va='top', rotation=90,
                fontsize=10, fontweight='bold', color=(r, g, b),
                fontfamily="monospace",
                transform=ax.get_xaxis_transform())
        ax.text(x_pos, y_pos - 0.06, first, ha='center', va='top', rotation=90,
                fontsize=10, fontweight='bold', color="dimgrey",
                fontfamily="monospace",
                transform=ax.get_xaxis_transform())
    
    ax.set_ylabel("Mean Mutations per Mb", fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    title = f"{file_path.stem.replace('_', ' ')} (Mean ± Std. Dev.)" if show_error_bars else file_path.stem.replace('_', ' ')
    ax.set_title(title)
    sns.despine(ax=ax, top=True, right=True)
    
    # Vertical separators between substitution groups
    group_types = [mut[2:5] for mut in mutation_types]
    prev_group = group_types[0]
    for i, grp in enumerate(group_types):
        if grp != prev_group:
            ax.axvline(i - 0.5, color='black', linewidth=1)
            prev_group = grp
    
    # Colored horizontal bars above the plot for each substitution group
    group_indices = {}
    for idx, grp in enumerate(group_types):
        group_indices.setdefault(grp, []).append(idx)
    
    y_max = channel_means.max()
    bar_height   = y_max * 0.05
    bar_y_bottom = y_max * 1.05
    bar_y_top    = bar_y_bottom + bar_height
    for grp, indices in group_indices.items():
        xmin = indices[0] - 0.5
        xmax = indices[-1] + 0.5
        width = xmax - xmin
        group_color = mutation_colors[grp]
        rect = patches.Rectangle((xmin, bar_y_bottom), width, bar_height,
                                 color=group_color, ec='none')
        ax.add_patch(rect)
        center = (xmin + xmax) / 2
        ax.text(center, bar_y_top + (bar_height * 0.2), grp,
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
    
    # ax.set_ylim(0, bar_y_top + y_max * 0.1)
    ax.set_ylim(0,120)
    plt.tight_layout()
    output_file = file_path.with_suffix('.96_channel_spectrum.svg')
    plt.savefig(output_file, format='svg', dpi=600)


def plot_96_channel_mutation_spectrum_with_complement_from_192_channels_tsv(file_path, show_error_bars=True):
    """
    Plot the mutation spectrum as 96 canonical channels shown as paired bars.
    
    The TSV file is assumed to contain 192 mutation-channel columns (i.e. separate counts for each strand).
    For each canonical channel:
      - The left bar shows the counts from the original (pyrimidine-centric) channels 
        (i.e. those where the reference base is C or T).
      - The right bar shows the counts from the complementary channels 
        (i.e. where the reference is A or G), converted into the canonical form.
        
    Both bars are labeled below (the left with the canonical label and the right with its complementary label).
    At the top, two horizontal boxes (one for original and one for complement) are drawn for each substitution group.
    
    The optional parameter `show_error_bars` toggles the display of error bars.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path, sep="\t")
    # Remove summary rows: drop rows where the first column is empty or equals "Mean"
    df_samples = df[~df["Unnamed: 0"].astype(str).str.strip().isin(["", "Mean"])].copy()
    
    # Identify all mutation-channel columns (excluding extras)
    excluded_cols = ["Unnamed: 0", "Genotype_Treatment", "Total"]
    mutation_cols = [col for col in df_samples.columns if col not in excluded_cols]
    df_samples[mutation_cols] = df_samples[mutation_cols].apply(pd.to_numeric)
    
    # Split the 192 channels into two groups.
    # For each column, extract the reference base (first character inside the bracket).
    orig_dict = {}  # canonical channel: list of columns that are already in pyrimidine (C/T) orientation
    comp_dict = {}  # canonical channel: list of columns that require conversion (ref is A or G)
    for col in mutation_cols:
        try:
            ref = col.split('[')[1][0]
        except Exception:
            continue
        canon = canonical_channel(col)
        if ref in "CT":
            orig_dict.setdefault(canon, []).append(col)
        else:
            comp_dict.setdefault(canon, []).append(col)
    
    # Define the standard order for the 96 canonical channels.
    bases = ['A', 'C', 'G', 'T']
    mutation_types = [f"{b1}[{ref}>{alt}]{b2}" 
                      for ref in 'CT' 
                      for alt in 'ACGT' if ref != alt 
                      for b1 in bases 
                      for b2 in bases]
    
    # For each canonical channel, build per-sample counts for original and complement.
    orig_series = {}
    comp_series = {}
    for mtype in mutation_types:
        if mtype in orig_dict:
            orig_series[mtype] = df_samples[orig_dict[mtype]].sum(axis=1)
        else:
            orig_series[mtype] = pd.Series(0, index=df_samples.index)
        if mtype in comp_dict:
            comp_series[mtype] = df_samples[comp_dict[mtype]].sum(axis=1)
        else:
            comp_series[mtype] = pd.Series(0, index=df_samples.index)
    
    orig_df = pd.DataFrame(orig_series)
    comp_df = pd.DataFrame(comp_series)
    
    # Compute per-channel means and standard deviations.
    orig_means = orig_df.mean()
    orig_stds  = orig_df.std()
    comp_means = comp_df.mean()
    comp_stds  = comp_df.std()
    
    # Define colors for each substitution type (based on the canonical channel’s mutation part).
    mutation_colors = {
        'C>A': (3/256, 189/256, 239/256),
        'C>G': (1/256, 1/256, 1/256),
        'C>T': (228/256, 41/256, 38/256),
        'T>A': (203/256, 202/256, 202/256),
        'T>C': (162/256, 207/256, 99/256),
        'T>G': (236/256, 199/256, 197/256)
    }
    # Original bar colors for the 96 channels.
    orig_colors = [mutation_colors[mt[2:5]] for mt in mutation_types]
    # Complementary bar colors: compute the complementary color for each original color.
    # comp_colors = [get_complementary_color(c) for c in orig_colors]
    comp_colors_dict = {
    'C>A': (0/256, 59/256, 109/256),   # Very dark blue
    'C>G': (70/256, 70/256, 70/256),   # dark gray
    'C>T': (98/256, 11/256, 8/256),    # Very dark red
    'T>A': (90/256, 90/256, 90/256), # Lighter gray (more distinct from black)
    'T>C': (52/256, 97/256, 9/256),    # Much darker green
    'T>G': (106/256, 69/256, 67/256)   # Much darker muted pink
    }

    comp_colors = [comp_colors_dict[mt[2:5]] for mt in mutation_types]
    # Compute the complementary channel labels.
    comp_labels = [complement_channel(mt) for mt in mutation_types]
    
    # --- Begin plotting ---
    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(30, 6))
    
    n_groups = len(mutation_types)  # 96
    x = np.arange(n_groups)
    bar_width = 0.4
    offset_left = -bar_width/2
    offset_right = bar_width/2
    
    # Plot original (left) bars.
    bars_orig = ax.bar(x + offset_left, orig_means.values, width=bar_width, 
                       yerr=orig_stds.values if show_error_bars else None, 
                       color=orig_colors, edgecolor='none', 
                       capsize=2, alpha=0.8, 
                       error_kw={'elinewidth': 0.5, 'ecolor': 'grey', 'capthick': 0.5} if show_error_bars else {})
    
    # Plot complementary (right) bars with complementary colors and no black border.
    bars_comp = ax.bar(x + offset_right, comp_means.values, width=bar_width, 
                       yerr=comp_stds.values if show_error_bars else None, 
                       color=comp_colors, edgecolor='none', 
                       capsize=2, alpha=0.8, 
                       error_kw={'elinewidth': 0.5, 'ecolor': 'grey', 'capthick': 0.5} if show_error_bars else {})
    
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.set_xticks([])
    
    # --- Add rotated labels below both bars ---
    # Labels for original bars.
    for i, label in enumerate(mutation_types):
        first = label[0]
        mid   = label[2]
        last  = label[-1]
        r, g, b = orig_colors[i]
        # Center label under the original bar.
        x_pos_orig = x[i] + offset_left + bar_width/2
        y_pos = -0.02
        ax.text(x_pos_orig, y_pos, last, ha='center', va='top', rotation=90, 
                fontsize=10, fontweight='bold', color="dimgrey", fontfamily="DejaVu Sans Mono",
                transform=ax.get_xaxis_transform())
        ax.text(x_pos_orig, y_pos - 0.02, mid, ha='center', va='top', rotation=90, 
                fontsize=10, fontweight='bold', color=(r, g, b), fontfamily="monospace",
                transform=ax.get_xaxis_transform())
        ax.text(x_pos_orig, y_pos - 0.04, first, ha='center', va='top', rotation=90, 
                fontsize=10, fontweight='bold', color="dimgrey", fontfamily="monospace",
                transform=ax.get_xaxis_transform())
    # Labels for complementary bars.
    for i, clabel in enumerate(comp_labels):
        if clabel is None:
            continue
        first = clabel[0]
        mid   = clabel[2]
        last  = clabel[-1]
        r_c, g_c, b_c = comp_colors[i]
        x_pos_comp = x[i] + offset_right + bar_width/2
        y_pos = -0.02
        ax.text(x_pos_comp, y_pos, last, ha='center', va='top', rotation=90, 
                fontsize=10, fontweight='bold', color="dimgrey", fontfamily="DejaVu Sans Mono",
                transform=ax.get_xaxis_transform())
        ax.text(x_pos_comp, y_pos - 0.02, mid, ha='center', va='top', rotation=90, 
                fontsize=10, fontweight='bold', color=(r_c, g_c, b_c), fontfamily="monospace",
                transform=ax.get_xaxis_transform())
        ax.text(x_pos_comp, y_pos - 0.04, first, ha='center', va='top', rotation=90, 
                fontsize=10, fontweight='bold', color="dimgrey", fontfamily="monospace",
                transform=ax.get_xaxis_transform())
    
    ax.set_ylabel("Mean Mutations per Genome")
    ax.set_title(f"{file_path.stem.replace('_', ' ')} 192 Channel Spectra", fontsize=14, pad=20)
    sns.despine(ax=ax, top=True, right=True)
    
    # --- Add vertical separators between substitution groups ---
    group_types = [mt[2:5] for mt in mutation_types]
    prev_group = group_types[0]
    for i, grp in enumerate(group_types):
        if grp != prev_group:
            ax.axvline(i - 0.5, color='black', linewidth=1)
            prev_group = grp
    
    # --- Add horizontal colored boxes at the top for substitution groups ---
    group_indices = {}
    for idx, grp in enumerate(group_types):
        group_indices.setdefault(grp, []).append(idx)

    y_max = max(orig_means.max(), comp_means.max())
    box_height = y_max * 0.05
    box_y_bottom = y_max * 1.15
    box_y_top = box_y_bottom + box_height
    group_label_fontsize = 14

    # Complementary group boxes (now on top)
    comp_ref = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}  # for computing complementary group label
    for grp, indices in group_indices.items():
        xmin = indices[0] - 0.5
        xmax = indices[-1] + 0.5
        width = xmax - xmin
        group_color_comp = comp_colors_dict[grp]  # Now using complementary group colors
        rect = patches.Rectangle((xmin, box_y_bottom), width, box_height, color=group_color_comp, ec='none')
        ax.add_patch(rect)
        center = (xmin + xmax) / 2
        # Compute the complementary substitution label. E.g., if grp is "C>A", then comp_grp becomes "G>T"
        comp_grp = f"{comp_ref[grp[0]]}>{comp_ref[grp[2]]}"
        ax.text(center, box_y_top + (box_height * 0.05), comp_grp,
                ha='center', va='bottom', fontsize=group_label_fontsize, fontweight='bold', color='black')

    # Original group boxes (now on bottom)
    comp_box_y_bottom = box_y_top + box_height * 1.3
    comp_box_y_top = comp_box_y_bottom + box_height
    for grp, indices in group_indices.items():
        xmin = indices[0] - 0.5
        xmax = indices[-1] + 0.5
        width = xmax - xmin
        group_color = mutation_colors[grp]  # Now using original group colors
        rect = patches.Rectangle((xmin, comp_box_y_bottom), width, box_height, color=group_color, ec='none')
        ax.add_patch(rect)
        center = (xmin + xmax) / 2
        ax.text(center, comp_box_y_top + (box_height * 0.05), grp,
                ha='center', va='bottom', fontsize=group_label_fontsize, fontweight='bold', color='black')
    
    ax.set_ylim(0, comp_box_y_top + box_height * 0.5)
    plt.tight_layout()
    # plt.show()
    output_file = file_path.with_suffix('.192_split_bar_plot.svg')
    plt.savefig(output_file, dpi=600)

def plot_96_channel_mutation_spectrum_with_complement_from_192_split_channels_tsv(file_path, show_error_bars=True):
    """
    Plot the mutation spectrum as two separate 96-channel subplots, both sharing the same y-axis:
      - Top subplot: counts from pyrimidine-centric channels (C/T).
      - Bottom subplot: counts from purine-centric channels (A/G),
        each converted into the same 96 canonical channels.

    The style, size, and coloring follow the same conventions as
    plot_combined_96_channel_mutation_spectrum_from_tsv:
      - Figure is 13 wide x 12 tall (stacked subplots).
      - Only left and bottom spines remain (despine top/right).
      - show_error_bars controls optional standard deviation bars.
      - Uses the same color palette and the same 96-channel ordering.

    Both subplots share a common y-axis range to make direct comparisons easier.

    Parameters
    ----------
    file_path : str or Path
        Path to the 192-channel .tsv file.
    show_error_bars : bool
        If True, include standard deviation error bars in each bar.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path, sep="\t")

    # Remove summary rows where first column is empty or "Mean"
    df_samples = df[~df["Unnamed: 0"].astype(str).str.strip().isin(["", "Mean"])].copy()

    # Identify mutation-channel columns (exclude known extras)
    excluded_cols = ["Unnamed: 0", "Genotype_Treatment", "Total"]
    mutation_cols = [col for col in df_samples.columns if col not in excluded_cols]
    df_samples[mutation_cols] = df_samples[mutation_cols].apply(pd.to_numeric)

    # Split columns by reference base: pyrimidine (C/T) vs. purine (A/G)
    orig_dict = {}
    comp_dict = {}
    for col in mutation_cols:
        try:
            ref = col.split('[')[1][0]
        except Exception:
            continue
        canon = canonical_channel(col)
        if ref in "CT":
            orig_dict.setdefault(canon, []).append(col)
        else:
            comp_dict.setdefault(canon, []).append(col)

    # Define the standard 96-channel order (pyrimidine-based)
    bases = ['A', 'C', 'G', 'T']
    mutation_types = [
        f"{b1}[{ref}>{alt}]{b2}"
        for ref in 'CT'
        for alt in 'ACGT' if ref != alt
        for b1 in bases
        for b2 in bases
    ]

    # Sum counts per sample for each canonical channel (orig vs comp)
    orig_series = {}
    comp_series = {}
    for mtype in mutation_types:
        if mtype in orig_dict:
            orig_series[mtype] = df_samples[orig_dict[mtype]].sum(axis=1)
        else:
            orig_series[mtype] = pd.Series(0, index=df_samples.index)

        if mtype in comp_dict:
            comp_series[mtype] = df_samples[comp_dict[mtype]].sum(axis=1)
        else:
            comp_series[mtype] = pd.Series(0, index=df_samples.index)

    orig_df = pd.DataFrame(orig_series)
    comp_df = pd.DataFrame(comp_series)

    # Compute means & std
    orig_means = orig_df.mean()
    orig_stds  = orig_df.std()
    comp_means = comp_df.mean()
    comp_stds  = comp_df.std()

    # --- Same color palette as plot_combined_96_channel_mutation_spectrum_from_tsv ---
    mutation_colors = {
        'C>A': [3/256, 189/256, 239/256],   # Light blue
        'C>G': [1/256, 1/256, 1/256],       # Black
        'C>T': [228/256, 41/256, 38/256],   # Red
        'T>A': [203/256, 202/256, 202/256], # Gray
        'T>C': [162/256, 207/256, 99/256],  # Green
        'T>G': [236/256, 199/256, 197/256]  # Salmon
    }

    # Top (pyrimidine) colors in the same 96-channel order
    orig_colors = [mutation_colors[m[2:5]] for m in mutation_types]

    # Complementary color dictionary (manually defined)
    comp_colors_dict = {
        'C>A': (  63/256,  0/256, 125/256),
        'C>G': ( 156/256, 117/256, 95/256),
        'C>T': ( 178/256, 31/256, 107/256),
        'T>A': ( 242/256, 142/256, 43/256),
        'T>C': ( 31/256, 158/256, 137/256),
        'T>G': (237/256, 201/256,  72/256),
    }
    # Bottom (purine) colors in the same channel order
    comp_colors = [comp_colors_dict[m[2:5]] for m in mutation_types]

    # Complementary channel labels
    comp_labels = [complement_channel(m) for m in mutation_types]

    # ----------------------------------
    # Define a single y-axis limit for both subplots
    # ----------------------------------
    if show_error_bars:
        max_orig = (orig_means + orig_stds).max()
        max_comp = (comp_means + comp_stds).max()
    else:
        max_orig = orig_means.max()
        max_comp = comp_means.max()

    # y_lim = max(max_orig, max_comp)

    # # hardcoded y limit
    y_lim = 300

    # We'll scale up a bit for boxes/text
    # y_axis_upper = y_lim * 1.3
    y_axis_upper = 300

    # -------------------------
    # Begin Plotting
    # -------------------------
    sns.set(style="white", context="talk")
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(14, 8),
        sharex=True, sharey=True  # share the same y-axis
    )

    x = np.arange(len(mutation_types))
    n_groups = len(mutation_types)
    bar_width = 0.8

    # =========================
    # TOP subplot: Pyrimidine
    # =========================
    ax_top.bar(
        x, orig_means.values,
        width=bar_width,
        yerr=orig_stds.values if show_error_bars else None,
        color=orig_colors,
        edgecolor='none',
        capsize=4,
        alpha=0.8,
        error_kw={'elinewidth':0.5, 'ecolor':'grey', 'capthick':0.5} if show_error_bars else {}
    )
    ax_top.set_xlim(-0.5, n_groups - 0.5)
    ax_top.set_ylim(0, y_axis_upper)
    ax_top.set_ylabel("Mean Mutations per Mb", fontsize=16)
    ax_top.tick_params(axis='y', labelsize=14)
    ax_top.set_title(f"{file_path.stem.replace('_', ' ')}\nPyrimidine-Based (Top)",
                     fontsize=16, pad=10)

    # Grid & despine
    ax_top.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax_top.set_axisbelow(True)
    ax_top.set_xticks([])
    sns.despine(ax=ax_top, top=True, right=True)

    # Vertical group separators
    group_types = [m[2:5] for m in mutation_types]
    prev_group = group_types[0]
    for i, grp in enumerate(group_types):
        if grp != prev_group:
            ax_top.axvline(i - 0.5, color='black', linewidth=1)
            prev_group = grp

    # Group-colored box at the top of the subplot
    group_indices = {}
    for idx, grp in enumerate(group_types):
        group_indices.setdefault(grp, []).append(idx)

    box_height = y_lim * 0.05
    box_y_bottom = y_lim * 1.05
    box_y_top = box_y_bottom + box_height

    for grp, indices in group_indices.items():
        xmin = indices[0] - 0.5
        xmax = indices[-1] + 0.5
        width = xmax - xmin
        grp_color = mutation_colors[grp]
        rect = patches.Rectangle((xmin, box_y_bottom), width, box_height, color=grp_color, ec='none')
        ax_top.add_patch(rect)
        center = (xmin + xmax) / 2
        ax_top.text(
            center, box_y_top + (box_height * 0.2),
            grp,
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='black'
        )

    # Rotated labels below each bar (canonical channel)
    for i, label in enumerate(mutation_types):
        first = label[0]
        mid   = label[2]
        last  = label[-1]
        (r, g, b) = orig_colors[i]
        x_pos = x[i]
        y_pos = -0.02
        ax_top.text(x_pos, y_pos, last,
                    ha='center', va='top', rotation=90,
                    fontsize=9, fontweight='bold', color="dimgrey",
                    transform=ax_top.get_xaxis_transform())
        ax_top.text(x_pos, y_pos - 0.03, mid,
                    ha='center', va='top', rotation=90,
                    fontsize=9, fontweight='bold', color=(r, g, b),
                    transform=ax_top.get_xaxis_transform())
        ax_top.text(x_pos, y_pos - 0.06, first,
                    ha='center', va='top', rotation=90,
                    fontsize=9, fontweight='bold', color="dimgrey",
                    transform=ax_top.get_xaxis_transform())

    # =========================
    # BOTTOM subplot: Purine
    # =========================
    ax_bottom.bar(
        x, comp_means.values,
        width=bar_width,
        yerr=comp_stds.values if show_error_bars else None,
        color=comp_colors,
        edgecolor='none',
        capsize=4,
        alpha=0.8,
        error_kw={'elinewidth':0.5, 'ecolor':'grey', 'capthick':0.5} if show_error_bars else {}
    )
    ax_bottom.set_xlim(-0.5, n_groups - 0.5)
    ax_bottom.set_ylim(0, y_axis_upper)
    ax_bottom.set_ylabel("Mean Mutations per Mb", fontsize=16)
    ax_bottom.tick_params(axis='y', labelsize=14)
    ax_bottom.set_title("Purine-Based (Bottom)", fontsize=16, pad=10)

    # Grid & despine
    ax_bottom.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax_bottom.set_axisbelow(True)
    ax_bottom.set_xticks([])
    sns.despine(ax=ax_bottom, top=True, right=True)

    # Vertical group separators
    prev_group = group_types[0]
    for i, grp in enumerate(group_types):
        if grp != prev_group:
            ax_bottom.axvline(i - 0.5, color='black', linewidth=1)
            prev_group = grp

    # Dictionary to map pyrimidine-based mutations to their complementary purine-based mutations
    comp_ref = {
        'C>A': 'G>T',
        'C>G': 'G>C',
        'C>T': 'G>A',
        'T>A': 'A>T',
        'T>C': 'A>G',
        'T>G': 'A>C'
    }

    # Group-colored box at the top of the bottom subplot (using complementary colors)
    for grp, indices in group_indices.items():
        xmin = indices[0] - 0.5
        xmax = indices[-1] + 0.5
        width = xmax - xmin
        i_first = indices[0]
        grp_color = comp_colors[i_first]  # color that matches the first channel of that group
        rect = patches.Rectangle((xmin, box_y_bottom), width, box_height, color=grp_color, ec='none')
        ax_bottom.add_patch(rect)
        center = (xmin + xmax) / 2
        # Compute the complementary substitution label (e.g., if grp is "C>A", then comp_grp becomes "G>T")
        comp_grp = comp_ref[grp]
        ax_bottom.text(
            center, box_y_top + (box_height * 0.2),
            comp_grp,  # Now displaying the complementary purine-based mutation
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='black'
        )

    # Rotated labels below each bar (complementary channel)
    for i, clabel in enumerate(comp_labels):
        if clabel is None:
            continue
        first = clabel[0]
        mid   = clabel[2]
        last  = clabel[-1]
        (r_c, g_c, b_c) = comp_colors[i]
        x_pos = x[i]
        y_pos = -0.02
        ax_bottom.text(x_pos, y_pos, last,
                       ha='center', va='top', rotation=90,
                       fontsize=9, fontweight='bold', color="dimgrey",
                       transform=ax_bottom.get_xaxis_transform())
        ax_bottom.text(x_pos, y_pos - 0.03, mid,
                       ha='center', va='top', rotation=90,
                       fontsize=9, fontweight='bold', color=(r_c, g_c, b_c),
                       transform=ax_bottom.get_xaxis_transform())
        ax_bottom.text(x_pos, y_pos - 0.06, first,
                       ha='center', va='top', rotation=90,
                       fontsize=9, fontweight='bold', color="dimgrey",
                       transform=ax_bottom.get_xaxis_transform())

    plt.tight_layout()
    # plt.show()
    output_name = file_path.with_suffix('.192_stacked_plots.svg')
    plt.savefig(output_name, format='svg', dpi=600)

def plot_96_channel_OE_split_subplots_from_192_tsv(
    observed_tsv: str | Path,
    expected_tsv: str | Path,
    show_error_bars: bool = True,
    pseudocount: float = 1e-6,
):
    """
    Plot log2-observed/expected (O/E) spectra as *stacked* pyrimidine (top) and
    purine (bottom) 96-channel subplots that share a common y-axis.

    This is a drop-in O/E version of `plot_96_channel_mutation_spectrum_with_
    complement_from_192_split_channels_tsv`, identical in look-and-feel except
    that bars now show **mean log₂(observed / expected)** per canonical channel.

    Parameters
    ----------
    observed_tsv : str | Path
        TSV containing observed 192-channel counts.
    expected_tsv : str | Path
        Matching TSV of expected counts (same columns / sample order).
    show_error_bars : bool, default True
        Draw ±1 SD error bars (per-sample log₂(O/E)).
    pseudocount : float, default 1e-6
        Added to every count before division to avoid log₂(0/0).
    """

    obs_path, exp_path = Path(observed_tsv), Path(expected_tsv)
    df_obs = pd.read_csv(obs_path, sep="\t")
    df_exp = pd.read_csv(exp_path, sep="\t")

    # ------------------------------------------------------------------ #
    # 1 . keep only sample rows
    mask = ~df_obs["Unnamed: 0"].astype(str).str.strip().isin(["", "Mean"])
    df_obs, df_exp = df_obs.loc[mask].copy(), df_exp.loc[mask].copy()

    excluded = {"Unnamed: 0", "Genotype_Treatment", "Total"}
    mut_cols = [c for c in df_obs.columns if c not in excluded]
    df_obs[mut_cols] = df_obs[mut_cols].apply(pd.to_numeric)
    df_exp[mut_cols] = df_exp[mut_cols].apply(pd.to_numeric)

    # ------------------------------------------------------------------ #
    # 2 . split columns by ref-base orientation
    orig_dict, comp_dict = {}, {}
    for col in mut_cols:
        try:
            ref = col.split('[')[1][0]
        except Exception:
            continue
        canon = canonical_channel(col)
        if ref in "CT":
            orig_dict.setdefault(canon, []).append(col)
        else:
            comp_dict.setdefault(canon, []).append(col)

    bases = ['A', 'C', 'G', 'T']
    mut96 = [
        f"{b1}[{ref}>{alt}]{b2}"
        for ref in "CT"
        for alt in "ACGT" if alt != ref
        for b1 in bases
        for b2 in bases
    ]

    # ------------------------------------------------------------------ #
    # 3 . per-sample log₂(O/E)
    def log2_oe(cols):
        o = df_obs[cols].sum(axis=1) + pseudocount
        e = df_exp[cols].sum(axis=1) + pseudocount
        return np.log2(o / e)

    orig_log2 = {m: log2_oe(orig_dict.get(m, [])) for m in mut96}
    comp_log2 = {m: log2_oe(comp_dict.get(m, [])) for m in mut96}
    orig_df, comp_df = pd.DataFrame(orig_log2), pd.DataFrame(comp_log2)

    orig_mean, orig_sd = orig_df.mean(), orig_df.std()
    comp_mean, comp_sd = comp_df.mean(), comp_df.std()

    # ------------------------------------------------------------------ #
    # 4 . colour maps (unchanged)
    mut_colour = {
        'C>A': (3/256, 189/256, 239/256),
        'C>G': (1/256, 1/256, 1/256),
        'C>T': (228/256, 41/256, 38/256),
        'T>A': (203/256, 202/256, 202/256),
        'T>C': (162/256, 207/256, 99/256),
        'T>G': (236/256, 199/256, 197/256),
    }
    comp_colour = {
        'C>A': (  63/256,  0/256, 125/256),
        'C>G': ( 156/256, 117/256, 95/256),
        'C>T': ( 178/256, 31/256, 107/256),
        'T>A': ( 242/256, 142/256, 43/256),
        'T>C': ( 31/256, 158/256, 137/256),
        'T>G': (237/256, 201/256,  72/256),
    }
    colours_orig = [mut_colour[m[2:5]]  for m in mut96]
    colours_comp = [comp_colour[m[2:5]] for m in mut96]
    comp_labels  = [complement_channel(m) for m in mut96]

    # ------------------------------------------------------------------ #
    # 5 . shared y-axis range
    if show_error_bars:
        max_orig = (orig_mean + orig_sd).max()
        max_comp = (comp_mean + comp_sd).max()
    else:
        max_orig, max_comp = orig_mean.max(), comp_mean.max()
    y_lim = max(max_orig, max_comp)
    y_upper = y_lim * 1.3

    # ------------------------------------------------------------------ #
    # 6 . plotting
    sns.set(style="white", context="talk")
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, sharey=True
    )

    x = np.arange(len(mut96))
    bw = 0.8

    # ---- TOP (pyrimidine) ----
    ax_top.bar(
        x, orig_mean, width=bw,
        yerr=orig_sd if show_error_bars else None,
        color=colours_orig, edgecolor='none', capsize=4, alpha=0.85,
        error_kw={'elinewidth':0.5,'ecolor':'grey','capthick':0.5} if show_error_bars else {}
    )
    ax_top.set_ylim(0, y_upper)
    ax_top.set_ylabel("Mean log₂(O / E)", fontsize=16)
    ax_top.set_title(
        f"{obs_path.stem.replace('_',' ')} vs {exp_path.stem.replace('_',' ')}\n"
        "Pyrimidine-based (top)", fontsize=16, pad=10
    )
    ax_top.yaxis.grid(True, linestyle='--', linewidth=0.5, color='grey')
    ax_top.set_axisbelow(True)
    ax_top.set_xticks([])
    sns.despine(ax=ax_top, top=True, right=True)

    # ---- BOTTOM (purine) ----
    ax_bot.bar(
        x, comp_mean, width=bw,
        yerr=comp_sd if show_error_bars else None,
        color=colours_comp, edgecolor='none', capsize=4, alpha=0.85,
        error_kw={'elinewidth':0.5,'ecolor':'grey','capthick':0.5} if show_error_bars else {}
    )
    ax_bot.set_ylim(0, y_upper)
    ax_bot.set_ylabel("Mean log₂(O / E)", fontsize=16)
    ax_bot.set_title("Purine-based (bottom)", fontsize=16, pad=10)
    ax_bot.yaxis.grid(True, linestyle='--', linewidth=0.5, color='grey')
    ax_bot.set_axisbelow(True)
    ax_bot.set_xticks([])
    sns.despine(ax=ax_bot, top=True, right=True)

    # ------------------------------------------------------------------ #
    # 7 . group separators & coloured boxes (shared code)
    group = [m[2:5] for m in mut96]
    idx_by_grp = {}
    for i, g in enumerate(group):
        idx_by_grp.setdefault(g, []).append(i)

    for ax in (ax_top, ax_bot):
        prev = group[0]
        for i, g in enumerate(group):
            if g != prev:
                ax.axvline(i-0.5, color='black', linewidth=1)
                prev = g

    box_h = y_lim * 0.05
    box_y0 = y_lim * 1.05
    box_y1 = box_y0 + box_h

    comp_ref = {'A':'T','T':'A','C':'G','G':'C'}
    # top boxes (pyrimidine colours)
    for g, idxs in idx_by_grp.items():
        xmin, xmax = idxs[0]-0.5, idxs[-1]+0.5
        rect = patches.Rectangle((xmin, box_y0), xmax-xmin, box_h,
                                 color=mut_colour[g], ec='none')
        ax_top.add_patch(rect)
        ax_top.text((xmin+xmax)/2, box_y1 + box_h*0.2, g,
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # bottom boxes (purine colours + comp labels)
    for g, idxs in idx_by_grp.items():
        xmin, xmax = idxs[0]-0.5, idxs[-1]+0.5
        rect = patches.Rectangle((xmin, box_y0), xmax-xmin, box_h,
                                 color=comp_colour[g], ec='none')
        ax_bot.add_patch(rect)
        cg = f"{comp_ref[g[0]]}>{comp_ref[g[2]]}"
        ax_bot.text((xmin+xmax)/2, box_y1 + box_h*0.2, cg,
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # ------------------------------------------------------------------ #
    # 8 . rotated base-triplet labels
    def add_triplet_labels(ax, colours, labels, offset=0):
        for i, lab in enumerate(labels):
            f, m, l = lab[0], lab[2], lab[-1]
            r, g, b = colours[i]
            y0 = -0.02 - offset
            ax.text(x[i], y0, l, ha='center', va='top', rotation=90,
                    fontsize=9, fontweight='bold', color='dimgrey',
                    transform=ax.get_xaxis_transform())
            ax.text(x[i], y0-0.03, m, ha='center', va='top', rotation=90,
                    fontsize=9, fontweight='bold', color=(r,g,b),
                    transform=ax.get_xaxis_transform())
            ax.text(x[i], y0-0.06, f, ha='center', va='top', rotation=90,
                    fontsize=9, fontweight='bold', color='dimgrey',
                    transform=ax.get_xaxis_transform())

    add_triplet_labels(ax_top, colours_orig, mut96, offset=0)
    add_triplet_labels(ax_bot, colours_comp, comp_labels, offset=0)

    plt.tight_layout()
    out_svg = obs_path.with_suffix('.log2_OE_stacked.svg')
    plt.savefig(out_svg, format='svg', dpi=600)
    # plt.show()
    return out_svg


def plot_comparison_96_channel_mutation_spectrum_from_tsv(
    file_path_1,
    file_path_2,
    mode='difference',
    eps=1e-9
):
    """
    Read two TSV files that each have 192 mutation channel columns (i.e. separate
    counts for each strand), combine complementary channels into 96 canonical channels
    for each file, and then compute either:
      - The difference (file2 - file1) in mean counts across samples (mode='difference')
      - The log2 fold change of file2 over file1 (mode='fold_change')
    Finally, plot the resulting 96-channel spectrum.

    Parameters
    ----------
    file_path_1 : str or Path
        Path to the first TSV file.
    file_path_2 : str or Path
        Path to the second TSV file.
    mode : {'difference', 'fold_change'}, default='difference'
        If 'difference', the bar heights are (mean2 - mean1).
        If 'fold_change', the bar heights are log2((mean2 + eps)/(mean1 + eps)).
    eps : float, default=1e-9
        Small constant to avoid division by zero in fold-change mode.
    """
    file_path_1 = Path(file_path_1)
    file_path_2 = Path(file_path_2)

    # --- Helper function to load, combine channels, and get mean across samples ---
    def load_and_get_96means(fp):
        df = pd.read_csv(fp, sep="\t")
        # Remove summary rows: drop rows where the first column is empty or equals "Mean"
        df_samples = df[~df["Unnamed: 0"].astype(str).str.strip().isin(["", "Mean"])].copy()
        
        # Identify the mutation channel columns (assumes these are not in excluded_cols)
        excluded_cols = ["Unnamed: 0", "Genotype_Treatment", "Total"]
        channel_cols = [col for col in df_samples.columns if col not in excluded_cols]
        
        # Make sure channel columns are numeric
        df_samples[channel_cols] = df_samples[channel_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Map each column name (192 channels) to its canonical (pyrimidine–centric) channel
        mapping = {col: canonical_channel(col) for col in channel_cols}
        
        # Group the columns by canonical channel and sum the counts per sample
        combined_dict = {}
        for col, canon in mapping.items():
            combined_dict.setdefault(canon, []).append(col)
        
        # Build a new DataFrame with one column per canonical channel
        df_canonical = pd.DataFrame(index=df_samples.index)
        for canon, cols in combined_dict.items():
            df_canonical[canon] = df_samples[cols].sum(axis=1)
        
        # Compute per-channel means across samples
        channel_means = df_canonical.mean()
        return channel_means

    channel_means_1 = load_and_get_96means(file_path_1)
    channel_means_2 = load_and_get_96means(file_path_2)

    # Define the standard 96-channel order
    bases = ['A', 'C', 'G', 'T']
    mutation_types = [
        f"{b1}[{ref}>{alt}]{b2}" 
        for ref in 'CT' 
        for alt in 'ACGT' if ref != alt 
        for b1 in bases 
        for b2 in bases
    ]
    
    # Ensure missing channels are zero
    for mtype in mutation_types:
        if mtype not in channel_means_1:
            channel_means_1[mtype] = 0
        if mtype not in channel_means_2:
            channel_means_2[mtype] = 0

    # Reindex in standard order
    channel_means_1 = channel_means_1.reindex(mutation_types)
    channel_means_2 = channel_means_2.reindex(mutation_types)
    
    # Compute difference or fold change
    if mode == 'difference':
        results = channel_means_2 - channel_means_1
        y_axis_label = "Difference in Mean Mutations per Mb\n(file2 - file1)"
        plot_suffix = "_difference"
    elif mode == 'fold_change':
        results = np.log2((channel_means_2 + eps) / (channel_means_1 + eps))
        y_axis_label = "Log2 Fold Change in Mean Mutations per Mb\n(log2(file2 / file1))"
        plot_suffix = "_log2fold"
    else:
        raise ValueError("Invalid mode. Must be 'difference' or 'fold_change'.")

    # Define colors for each substitution type (the 6 possible ref>alt across pyrimidines)
    mutation_colors = {
        'C>A': [3/256, 189/256, 239/256],   # Light blue
        'C>G': [1/256, 1/256, 1/256],      # Black
        'C>T': [228/256, 41/256, 38/256],  # Red
        'T>A': [203/256, 202/256, 202/256],# Grey
        'T>C': [162/256, 207/256, 99/256], # Green
        'T>G': [236/256, 199/256, 197/256] # Salmon
    }
    colors = [mutation_colors[m[2:5]] for m in mutation_types]

    # --- Begin plotting ---
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(mutation_types))
    ax.bar(x, results.values, color=colors, edgecolor='none', alpha=0.8)
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlim(-0.5, len(mutation_types) - 0.5)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.set_xticks([])
    
    # Label the x-axis with the 3-letter contexts
    for i, label in enumerate(mutation_types):
        first = label[0]
        mid   = label[2]
        last  = label[-1]
        r, g, b = colors[i]
        x_position = x[i]
        # slightly shifting the text positions
        y_position = -0.02
        ax.text(x_position, y_position, last, ha='center', va='top', rotation=90,
                fontsize=10, fontweight='bold', color="dimgrey", fontfamily="DejaVu Sans Mono",
                transform=ax.get_xaxis_transform())
        ax.text(x_position, y_position - 0.03, mid, ha='center', va='top', rotation=90,
                fontsize=10, fontweight='bold', color=(r, g, b), fontfamily="monospace",
                transform=ax.get_xaxis_transform())
        ax.text(x_position, y_position - 0.06, first, ha='center', va='top', rotation=90,
                fontsize=10, fontweight='bold', color="dimgrey", fontfamily="monospace",
                transform=ax.get_xaxis_transform())

    ax.set_ylabel(y_axis_label, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)

    # Build vertical separators between substitution groups
    group_types = [m[2:5] for m in mutation_types]
    prev_group = group_types[0]
    for i, grp in enumerate(group_types):
        if grp != prev_group:
            ax.axvline(i - 0.5, color='black', linewidth=1)
            prev_group = grp
    
    # Add colored horizontal bars above the plot for each substitution group
    group_indices = {}
    for idx, grp in enumerate(group_types):
        group_indices.setdefault(grp, []).append(idx)
    
    # We'll place the group bars just above the current top
    y_min, y_max = ax.get_ylim()
    bar_height = (y_max - y_min) * 0.05
    bar_y_bottom = y_max + (y_max - y_min) * 0.02
    bar_y_top = bar_y_bottom + bar_height
    
    for grp, indices in group_indices.items():
        xmin = indices[0] - 0.5
        xmax = indices[-1] + 0.5
        width = xmax - xmin
        group_color = mutation_colors[grp]
        rect = patches.Rectangle((xmin, bar_y_bottom), width, bar_height,
                                 color=group_color, ec='none')
        ax.add_patch(rect)
        center = (xmin + xmax) / 2
        ax.text(center, bar_y_top + (bar_height * 0.2), grp,
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    # Adjust the plot limits so the top bar is visible
    ax.set_ylim(y_min, bar_y_top + (bar_height * 0.7))

    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    
    # Construct a filename to save
    out_name = f"{file_path_1.stem}_vs_{file_path_2.stem}{plot_suffix}.svg"
    out_name = file_path_1.parent / out_name
    plt.savefig(out_name, format='svg', dpi=600)
    # plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from pathlib import Path

# If you use Seaborn styling, import here; otherwise you can omit
import seaborn as sns

##########################
# Helper: canonical_channel
##########################
def canonical_channel(channel):
    """
    Convert a channel like X[Y>Z]W into pyrimidine-centric format.
    E.g., 'T[G>T]T' -> 'A[C>A]A'. If Y is already C or T, leave it alone.
    """
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    try:
        first = channel[0]
        inside = channel[channel.find('[') + 1 : channel.find(']')]
        ref, alt = inside.split('>')
        last = channel[-1]
        context = first + ref + last
        
        if ref in 'AG':
            # Reverse complement
            new_context = "".join(comp[b] for b in reversed(context))
            new_ref = 'T' if ref == 'A' else 'C'
            if alt == 'T':
                new_alt = 'A'
            elif alt == 'C':
                new_alt = 'G'
            elif alt == 'G':
                new_alt = 'C'
            elif alt == 'A':
                new_alt = 'T'
            else:
                new_alt = alt
            return f"{new_context[0]}[{new_ref}>{new_alt}]{new_context[2]}"
        else:
            return channel
    except Exception:
        return channel


##########################
# Helper: complement_channel
##########################
def complement_channel(channel):
    """
    Given a canonical pyrimidine‐centric channel (e.g. 'A[C>A]A'),
    return its purine‐based complement (e.g. 'T[G>T]T').
    """
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    try:
        first = channel[0]
        inside = channel[channel.find('[') + 1 : channel.find(']')]
        ref, alt = inside.split('>')
        last = channel[-1]
        # Reverse complement the 3-letter context
        new_context = "".join(comp[b] for b in reversed(first + ref + last))
        # Convert ref & alt
        mapping = {'C':'G', 'T':'A', 'G':'C', 'A':'T'}
        new_ref = mapping.get(ref, ref)
        new_alt = mapping.get(alt, alt)
        return f"{new_context[0]}[{new_ref}>{new_alt}]{new_context[2]}"
    except Exception:
        return None


###################################################################
# NEW FUNCTION: Plot difference or fold change in stacked subplots
###################################################################
def plot_comparison_96_channel_mutation_spectrum_stacked_from_192_split_channels_tsv(
    file_path_1,
    file_path_2,
    mode='difference',
    eps=1e-9
):
    """
    Read two 192-channel .tsv files (each with separate columns for forward/reverse),
    split channels into pyrimidine-centric vs. purine-based sets,
    then compute the difference (file2 - file1) or the log2 fold change
    (log2(file2 / file1)) for both sets.

    Produces two stacked subplots:
      - Top: Pyrimidine-based difference/fold-change
      - Bottom: Purine-based difference/fold-change
    Both subplots share the same X axis (the 96 canonical channels)
    and a consistent Y axis range for easier comparison.

    Parameters
    ----------
    file_path_1 : str or Path
        Path to the first 192-channel .tsv
    file_path_2 : str or Path
        Path to the second 192-channel .tsv
    mode : {'difference', 'fold_change'}, default='difference'
        If 'difference', compute (mean2 - mean1).
        If 'fold_change', compute log2( (mean2 + eps)/(mean1 + eps) ).
    eps : float, default=1e-9
        Small constant for division to avoid log(0).
    """
    file_path_1 = Path(file_path_1)
    file_path_2 = Path(file_path_2)

    # -----------------------------
    # 1) Load each file, parse columns
    # -----------------------------
    def load_pyrimidine_and_purine_means(fp):
        df = pd.read_csv(fp, sep="\t")

        # Remove summary rows: first column empty or equals "Mean"
        df_samples = df[~df["Unnamed: 0"].astype(str).str.strip().isin(["", "Mean"])].copy()

        # Identify mutation columns
        excluded_cols = ["Unnamed: 0", "Genotype_Treatment", "Total"]
        mutation_cols = [c for c in df_samples.columns if c not in excluded_cols]

        df_samples[mutation_cols] = df_samples[mutation_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Separate columns into pyrimidine-based vs purine-based by checking reference base
        # We'll sum them into canonical-channel keys
        pyrim_dict = {}  # sum of columns that have ref in "CT"
        purine_dict = {} # sum of columns that have ref in "AG"

        for col in mutation_cols:
            try:
                ref_base = col.split('[')[1][0]
            except Exception:
                continue
            canon = canonical_channel(col)
            if ref_base in "CT":
                pyrim_dict.setdefault(canon, []).append(col)
            else:
                purine_dict.setdefault(canon, []).append(col)

        # Create standard 96-channel order
        bases = ['A', 'C', 'G', 'T']
        mutation_types = [
            f"{b1}[{ref}>{alt}]{b2}"
            for ref in 'CT'
            for alt in 'ACGT' if ref != alt
            for b1 in bases
            for b2 in bases
        ]

        # For each canonical channel in the standard 96, sum up across relevant columns
        pyrim_df = pd.DataFrame(index=df_samples.index)
        purine_df = pd.DataFrame(index=df_samples.index)
        for mtype in mutation_types:
            if mtype in pyrim_dict:
                pyrim_df[mtype] = df_samples[pyrim_dict[mtype]].sum(axis=1)
            else:
                pyrim_df[mtype] = 0
            if mtype in purine_dict:
                purine_df[mtype] = df_samples[purine_dict[mtype]].sum(axis=1)
            else:
                purine_df[mtype] = 0

        # Compute means (no error bars in this difference/fold-change version)
        pyrim_means = pyrim_df.mean()
        purine_means = purine_df.mean()

        return pyrim_means, purine_means, mutation_types

    # Load each file
    pyrim_means_1, purine_means_1, mutation_types = load_pyrimidine_and_purine_means(file_path_1)
    pyrim_means_2, purine_means_2, _            = load_pyrimidine_and_purine_means(file_path_2)

    # -----------------------------
    # 2) Compute difference or fold change
    # -----------------------------
    if mode == 'difference':
        pyrim_results = pyrim_means_2 - pyrim_means_1
        purine_results = purine_means_2 - purine_means_1
        y_axis_label = "Difference (file2 - file1)"
        suffix = "_difference"
    elif mode == 'fold_change':
        pyrim_results = np.log2((pyrim_means_2 + eps) / (pyrim_means_1 + eps))
        purine_results = np.log2((purine_means_2 + eps) / (purine_means_1 + eps))
        y_axis_label = "Log2 Fold Change (file2 / file1)"
        suffix = "_log2FC"
    else:
        raise ValueError("mode must be either 'difference' or 'fold_change'.")

    # -----------------------------
    # 3) Plotting: top = pyrimidine, bottom = purine
    # -----------------------------
    # Colors for the 6 categories
    mutation_colors = {
        'C>A': [3/256, 189/256, 239/256],   # Light blue
        'C>G': [1/256, 1/256, 1/256],       # Black
        'C>T': [228/256, 41/256, 38/256],   # Red
        'T>A': [203/256, 202/256, 202/256], # Gray
        'T>C': [162/256, 207/256, 99/256],  # Green
        'T>G': [236/256, 199/256, 197/256]  # Salmon
    }
    # For the top (pyrimidine) subplot
    top_colors = [mutation_colors[m[2:5]] for m in mutation_types]

    # For the bottom (purine) subplot, define complementary colors if you like,
    # or just reuse the same dictionary. For clarity, let's define a slightly
    # darker version or a separate scheme:
    comp_ref_map = {'C>A':'G>T','C>G':'G>C','C>T':'G>A','T>A':'A>T','T>C':'A>G','T>G':'A>C'}
    purine_colors_dict = {
        'C>A': (  63/256,  0/256, 125/256),
        'C>G': ( 156/256, 117/256, 95/256),
        'C>T': ( 178/256, 31/256, 107/256),
        'T>A': ( 242/256, 142/256, 43/256),
        'T>C': ( 31/256, 158/256, 137/256),
        'T>G': (237/256, 201/256,  72/256),
    }
    bottom_colors = []
    for m in mutation_types:
        # m[2:5] is something like "C>A", "C>T", etc.
        # If we want the 'complement' label, we can do:
        # complement_label = comp_ref_map[m[2:5]]  # e.g. "C>A" -> "G>T"
        # color = purine_colors_dict[complement_label] ...
        # But usually we can just do:
        color = purine_colors_dict.get(m[2:5], (0.5,0.5,0.5))
        bottom_colors.append(color)

    x = np.arange(len(mutation_types))

    # Overall figure with two subplots
    sns.set(style="white", context="talk")
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(14, 8),
        sharex=True,  # same x ticks
        sharey=True   # same y range
    )

    # --- Plot data for top subplot (pyrimidine) ---
    ax_top.bar(
        x, pyrim_results.values,
        width=0.8,
        color=top_colors,
        edgecolor='none',
        alpha=0.8
    )
    # Horizontal reference line at 0 for difference/fold-change
    ax_top.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax_top.set_ylabel(y_axis_label, fontsize=14)
    ax_top.tick_params(axis='y', labelsize=12)
    ax_top.set_title(
        f"{file_path_1.stem} vs {file_path_2.stem}\nPyrimidine-based (top)",
        fontsize=15, pad=10
    )
    ax_top.set_xlim(-0.5, len(mutation_types) - 0.5)

    # Add vertical lines between mutation groups
    group_types = [m[2:5] for m in mutation_types]
    prev_group = group_types[0]
    for i, grp in enumerate(group_types):
        if grp != prev_group:
            ax_top.axvline(i - 0.5, color='black', linewidth=1)
            prev_group = grp

    ax_top.xaxis.set_ticks([])  # We won't show any x ticks at the top
    ax_top.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax_top.set_axisbelow(True)
    sns.despine(ax=ax_top, top=True, right=True)

    # Group-colored box above the top subplot
    group_indices = {}
    for idx, grp in enumerate(group_types):
        group_indices.setdefault(grp, []).append(idx)

    y_min_top, y_max_top = ax_top.get_ylim()
    bar_height = (y_max_top - y_min_top) * 0.05
    bar_y_bottom = y_max_top + (y_max_top - y_min_top) * 0.02
    bar_y_top = bar_y_bottom + bar_height
    for grp, indices in group_indices.items():
        xmin = indices[0] - 0.5
        xmax = indices[-1] + 0.5
        w = xmax - xmin
        grp_color = mutation_colors[grp]
        rect = patches.Rectangle((xmin, bar_y_bottom), w, bar_height, color=grp_color, ec='none')
        ax_top.add_patch(rect)
        center = (xmin + xmax) / 2
        ax_top.text(
            center,
            bar_y_top + (bar_height * 0.2),
            grp,
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
    # Adjust top subplot y-limit for the color bars
    ax_top.set_ylim(y_min_top, bar_y_top + bar_height * 0.7)

    # Label each bar with the 3-letter canonical context
    for i, label in enumerate(mutation_types):
        first, ref, last = label[0], label[2], label[-1]
        r, g, b = top_colors[i]
        x_pos = x[i]
        y_pos = -0.02
        ax_top.text(
            x_pos, y_pos, last,
            ha='center', va='top', rotation=90,
            fontsize=9, fontweight='bold', color="dimgrey",
            transform=ax_top.get_xaxis_transform()
        )
        ax_top.text(
            x_pos, y_pos - 0.03, ref,
            ha='center', va='top', rotation=90,
            fontsize=9, fontweight='bold', color=(r, g, b),
            transform=ax_top.get_xaxis_transform()
        )
        ax_top.text(
            x_pos, y_pos - 0.06, first,
            ha='center', va='top', rotation=90,
            fontsize=9, fontweight='bold', color="dimgrey",
            transform=ax_top.get_xaxis_transform()
        )

    # --- Plot data for bottom subplot (purine) ---
    ax_bottom.bar(
        x, purine_results.values,
        width=0.8,
        color=bottom_colors,
        edgecolor='none',
        alpha=0.8
    )
    ax_bottom.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_bottom.set_ylabel(y_axis_label, fontsize=14)
    ax_bottom.tick_params(axis='y', labelsize=12)
    ax_bottom.set_title("Purine-based (bottom)", fontsize=15, pad=10)
    ax_bottom.set_xlim(-0.5, len(mutation_types) - 0.5)

    # Same group separators
    prev_group = group_types[0]
    for i, grp in enumerate(group_types):
        if grp != prev_group:
            ax_bottom.axvline(i - 0.5, color='black', linewidth=1)
            prev_group = grp

    ax_bottom.xaxis.set_ticks([])
    ax_bottom.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax_bottom.set_axisbelow(True)
    sns.despine(ax=ax_bottom, top=True, right=True)

    # Group-colored box above the bottom subplot
    y_min_bot, y_max_bot = ax_bottom.get_ylim()
    # Possibly unify these with the top if you want truly "shared" y-lims
    # But we already set sharey=True, so let's re-check after the bottom bars are drawn
    y_min, y_max = ax_top.get_ylim()
    ax_bottom.set_ylim(y_min, y_max)  # unify top & bottom range

    # We add the color boxes again in the same region as the top's region
    for grp, indices in group_indices.items():
        xmin = indices[0] - 0.5
        xmax = indices[-1] + 0.5
        w = xmax - xmin
        # We'll pick the "bottom_colors" for each group, or do something custom:
        i_first = indices[0]
        rect_color = bottom_colors[i_first]
        rect = patches.Rectangle((xmin, bar_y_bottom), w, bar_height, color=rect_color, ec='none')
        ax_bottom.add_patch(rect)
        center = (xmin + xmax) / 2
        # If you want to show the complementary mutation label, e.g. "C>A" -> "G>T"
        # we can do:
        comp_label = comp_ref_map[group_types[i_first]]  # e.g. if group_types[i_first] is 'C>A', comp_label = 'G>T'
        ax_bottom.text(
            center,
            bar_y_top + (bar_height * 0.2),
            comp_label,
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    # Label each bar with the complementary channel
    # If you prefer, you could label them the same as the top, or skip labels entirely
    comp_labels = [complement_channel(m) for m in mutation_types]
    for i, c_label in enumerate(comp_labels):
        if not c_label:
            continue
        first, ref, last = c_label[0], c_label[2], c_label[-1]
        r, g, b = bottom_colors[i]
        x_pos = x[i]
        y_pos = -0.02
        ax_bottom.text(
            x_pos, y_pos, last,
            ha='center', va='top', rotation=90,
            fontsize=9, fontweight='bold', color="dimgrey",
            transform=ax_bottom.get_xaxis_transform()
        )
        ax_bottom.text(
            x_pos, y_pos - 0.03, ref,
            ha='center', va='top', rotation=90,
            fontsize=9, fontweight='bold', color=(r, g, b),
            transform=ax_bottom.get_xaxis_transform()
        )
        ax_bottom.text(
            x_pos, y_pos - 0.06, first,
            ha='center', va='top', rotation=90,
            fontsize=9, fontweight='bold', color="dimgrey",
            transform=ax_bottom.get_xaxis_transform()
        )

    plt.tight_layout()

    # Construct a filename to save
    out_name = f"{file_path_1.stem}_vs_{file_path_2.stem}{suffix}_stacked.svg"
    out_path = file_path_1.parent / out_name
    plt.savefig(out_path, format='svg', dpi=600)
    # plt.show()
