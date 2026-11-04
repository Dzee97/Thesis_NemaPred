import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
from common import parse_config
from matplotlib.axes import Axes
from scipy.stats import entropy
from tqdm import tqdm


@dataclass
class Config:
    input_metadata: Path
    input_18s_ref_fasta: Path
    input_18s_otu_fasta: Path
    input_28s_ref_fasta: Path
    input_coi_ref_and_otu_prot_fasta: Path
    input_coi_ref_trimmed_fasta: Path
    input_coi_otu_trimmed_fasta: Path
    output_fasta: Path
    output_metadata: Path


def create_minimal_msa(
    seq_dict: dict[str, skbio.RNA | skbio.Protein],
    nuc_seqs: None | dict[str, skbio.DNA] = None,
) -> tuple[dict[str, int], np.ndarray]:

    alignment_length = -1
    unique_positions = set()

    for seq in seq_dict.values():
        if alignment_length < 0:
            alignment_length = len(seq)

        non_gap_mask = ~seq.gaps()
        positions = np.flatnonzero(non_gap_mask).astype(np.uint32)
        unique_positions |= set(positions)

    unique_positions = np.sort(list(unique_positions))
    new_position_map = np.full(alignment_length, -1, dtype=np.int32)
    new_position_map[unique_positions] = np.arange(len(unique_positions))

    msa_index = {}

    msa_length = len(unique_positions) if nuc_seqs is None else len(unique_positions) * 3
    msa_matrix = np.full((len(seq_dict), msa_length), ord("-"), dtype=np.uint8)

    for i, (accession, seq) in enumerate(seq_dict.items()):
        msa_index[accession] = i

        non_gap_mask = ~seq.gaps()
        new_positions = new_position_map[non_gap_mask]

        if nuc_seqs is not None:
            new_positions = new_positions * 3
            new_positions = np.sort(
                np.concatenate((new_positions, new_positions + 1, new_positions + 2))
            )

            nuc_seq = nuc_seqs[accession]
        else:
            nuc_seq = seq

        ungapped_seq = nuc_seq.degap().values.view(np.uint8)
        msa_matrix[i, new_positions] = ungapped_seq

    return msa_index, msa_matrix


# --- Global worker variables (initialized once per process) ---
_WORKER_DF = None
_WORKER_MSA = None
_WORKER_PARAMS = {}


def _init_worker(df: pd.DataFrame, msa_matrix: np.ndarray, params: dict) -> None:
    """Initializes read-only shared data in each worker process."""
    global _WORKER_DF, _WORKER_MSA, _WORKER_PARAMS
    _WORKER_DF = df
    _WORKER_MSA = msa_matrix
    _WORKER_PARAMS = params


def _process_column(col_idx: int) -> tuple[int, dict[str, float] | None]:
    """Processes a single MSA column inside a worker process."""
    df = cast(pd.DataFrame, _WORKER_DF)
    msa_matrix = cast(np.ndarray, _WORKER_MSA)
    min_observations: int = _WORKER_PARAMS["min_observations"]
    min_global_entropy: float = _WORKER_PARAMS["min_global_entropy"]
    min_split_size: int = _WORKER_PARAMS["min_split_size"]
    tax_levels: list[str] = _WORKER_PARAMS["tax_levels"]

    def calculate_entropy(elements: np.ndarray) -> float:
        _, counts = np.unique(elements, return_counts=True)
        probs = counts / len(elements)
        return float(entropy(probs, base=2))

    col_vals = msa_matrix[:, col_idx]
    nondegen_mask = np.isin(col_vals, [ord(c) for c in ["A", "C", "G", "T"]])

    df_nondegen = df[nondegen_mask]
    is_species = df_nondegen.worms_rank == "Species"

    rep_mask = np.zeros_like(nondegen_mask)
    rep_mask[df_nondegen.index.to_numpy()] = ~is_species.to_numpy()

    for _, sp_rows in df_nondegen[is_species].groupby("worms_name", sort=False):
        sp_col_vals = col_vals[sp_rows.index.to_numpy()]

        vals, counts = np.unique(sp_col_vals, return_counts=True)
        max_idx = np.argmax(counts)
        max_val = vals[max_idx]

        rep_sp_idx = np.argmax(sp_col_vals == max_val)
        rep_all_idx = sp_rows.index[int(rep_sp_idx)]

        rep_mask[rep_all_idx] = True

    if np.sum(rep_mask) < min_observations:
        return col_idx, None

    df_rep = df[rep_mask]
    global_entropy = calculate_entropy(col_vals[rep_mask])

    if global_entropy < min_global_entropy:
        return col_idx, None

    level_entropy = {"global": global_entropy}

    for i, level in enumerate(tax_levels):
        group_entropies = []
        group_weights = []
        other_bucket = []

        for _, group_rows in df_rep.groupby(f"worms_{level}", sort=False):
            if i > 0 and len(group_rows) < min_split_size:
                other_bucket.extend(group_rows.index.to_list())
                continue

            group_idx = group_rows.index.to_numpy()
            group_entropies.append(calculate_entropy(col_vals[group_idx]))
            group_weights.append(len(group_rows) / len(df_rep))

        if i > 1 and len(other_bucket) > 0:
            tax_step = 1
            j = i - tax_step  # Fixed assignment typo from original snippet
            while j >= 0 and len(other_bucket) > 0:
                higher_level = tax_levels[j]
                df_other = df_rep.loc[other_bucket]
                other_bucket = []
                for _, group_rows in df_other.groupby(f"worms_{higher_level}", sort=False):
                    if j > 0 and len(group_rows) < min_split_size:
                        other_bucket.extend(group_rows.index.to_list())
                        continue

                    group_idx = group_rows.index.to_numpy()
                    group_entropies.append(calculate_entropy(col_vals[group_idx]))
                    group_weights.append(len(group_rows) / len(df_rep))

                tax_step += 1
                j = i - tax_step

        level_entropy[level] = float(np.average(group_entropies, weights=group_weights))

    return col_idx, level_entropy


def calculate_species_balanced_column_entropies(
    df: pd.DataFrame,
    msa_matrix: np.ndarray,
    tax_levels: list[str],
    min_observations=20,
    min_global_entropy=0.25,
    min_split_size=3,
    num_processes=None,
) -> dict[str, np.ndarray]:
    msa_length = msa_matrix.shape[1]

    col_level_entropy = {l: np.full(msa_length, np.nan) for l in ["global"] + tax_levels}

    params = {
        "min_observations": min_observations,
        "min_global_entropy": min_global_entropy,
        "min_split_size": min_split_size,
        "tax_levels": tax_levels,
    }

    # Set up multiprocessing Pool with initializer
    with multiprocessing.Pool(
        initializer=_init_worker, initargs=(df, msa_matrix, params), processes=num_processes
    ) as pool:
        # Use imap wrapped inside tqdm with a chunksize for better throughput
        results = tqdm(
            pool.imap(_process_column, range(msa_length), chunksize=10),
            total=msa_length,
            desc="Calculating column entropies",
        )

        for col_idx, level_entropy in results:
            if level_entropy is None:
                continue
            for level in level_entropy:
                col_level_entropy[level][col_idx] = level_entropy[level]

    return col_level_entropy


def plot_entropies(ref_entropies, ref_coverage, outfile, otu_entropies=None, otu_coverage=None):
    """
    Plots column entropies in sequential order and as a distribution.
    """

    x = np.arange(len(ref_entropies))

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 1. Ordered Line Plot
    ax1.scatter(
        x,
        ref_entropies,
        c="steelblue",
        alpha=0.8,
        s=6.0,
        label="NCBI Refs",
    )
    if otu_entropies is not None:
        ax1.scatter(
            x,
            otu_entropies,
            c="coral",
            alpha=0.8,
            s=6.0,
            label="OTUs",
        )
    ax1.set_title("Column Entropies (Alignment Order)")
    ax1.set_xlabel("Alignment Columns")
    ax1.set_ylabel("Shannon Entropy (bits)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # 2. Distribution Histogram
    # Automatically determine bins using 'auto', or use a fixed integer like 15
    ax2.scatter(
        x,
        ref_coverage,
        c="steelblue",
        alpha=0.8,
        s=6.0,
        label="NCBI Refs",
    )
    if otu_coverage is not None:
        ax2.scatter(
            x,
            otu_coverage,
            c="coral",
            alpha=0.8,
            s=6.0,
            label="OTUs",
        )
    ax2.set_title("Column Coverage (Alignment Order)")
    ax2.set_xlabel("Alignment Columns")
    ax2.set_ylabel("Relative Frequency")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    # Clean up layout and display
    plt.tight_layout()
    plt.savefig(outfile)


def plot_entropy_overlap_histogram(entropy_overlap_matrix, outfile, max_samples=100_000, seed=42):
    """Creates a sampled histogram of the entropy-weighted overlap scores

    for all unique pairwise sequence comparisons.
    """
    # 1. Get upper triangle indices (unique pairs, excluding the diagonal)
    row_indices, col_indices = np.triu_indices_from(entropy_overlap_matrix, k=1)
    total_pairs = len(row_indices)

    # 2. Randomly sample if the dataset is large
    if total_pairs > max_samples:
        np.random.seed(seed)  # For reproducibility
        sampled_idx = np.random.choice(total_pairs, size=max_samples, replace=False)
        y_vals = entropy_overlap_matrix[row_indices[sampled_idx], col_indices[sampled_idx]]
        print(f"Sampling {max_samples:,} pairs out of {total_pairs:,} total for fast plotting.")
    else:
        y_vals = entropy_overlap_matrix[row_indices, col_indices]
        print(f"Plotting all {total_pairs:,} pairs.")

    # 3. Set up and draw the histogram
    plt.figure(figsize=(8, 5))
    plt.hist(y_vals, bins=50, color="#2b5c8f", edgecolor="black", alpha=0.7)

    # 4. Styling and labels
    plt.title(
        f"Distribution of Entropy-Weighted Overlaps ({len(y_vals):,} sampled pairs)",
        fontsize=12,
    )
    plt.xlabel("Entropy-Weighted Overlap Score (Sum of Bit Entropies)", fontsize=10)
    plt.ylabel("Frequency (Number of Sequence Pairs)", fontsize=10)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(outfile)


def plot_entropy_overlap_cdf(entropy_overlap_matrix, outfile, max_samples=100_000, seed=42):
    """Creates a sampled Cumulative Distribution Function (CDF) plot

    of the entropy-weighted overlap scores for unique pairwise sequence comparisons.
    """
    # 1. Get upper triangle indices (unique pairs, excluding the diagonal)
    row_indices, col_indices = np.triu_indices_from(entropy_overlap_matrix, k=1)
    total_pairs = len(row_indices)

    # 2. Randomly sample if the dataset is large
    if total_pairs > max_samples:
        np.random.seed(seed)  # For reproducibility
        sampled_idx = np.random.choice(total_pairs, size=max_samples, replace=False)
        scores = entropy_overlap_matrix[row_indices[sampled_idx], col_indices[sampled_idx]]
        print(f"Sampling {max_samples:,} pairs out of {total_pairs:,} total for CDF plotting.")
    else:
        scores = entropy_overlap_matrix[row_indices, col_indices]
        print(f"Plotting all {total_pairs:,} pairs.")

    # 3. Compute empirical CDF values
    sorted_scores = np.sort(scores)
    cdf_vals = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    # 4. Set up and draw the plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        sorted_scores,
        cdf_vals,
        color="#2b5c8f",
        linewidth=2.5,
        label="Empirical CDF",
    )

    # 5. Styling and labels
    plt.title(
        f"Cumulative Distribution of Entropy Overlaps ({len(scores):,} sampled pairs)",
        fontsize=12,
    )
    plt.xlabel("Entropy-Weighted Overlap Score (Sum of Bit Entropies)", fontsize=10)
    plt.ylabel("Cumulative Proportion of Pairs", fontsize=10)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.legend(loc="lower right")

    plt.savefig(outfile)


def plot_entropy_boxplots(gene_entropy_dicts, gene_colors, outfile, figsize=(10, 6)):
    """Plots side-by-side box plots comparing 1D entropy arrays

    mapped from a dictionary of string labels.
    """

    fig, axs = plt.subplots(1, len(gene_entropy_dicts), figsize=figsize, sharey=True)  # type: ignore
    axs: list[Axes] = list(axs)

    for i, ((gene, entropy_dict), color) in enumerate(zip(gene_entropy_dicts.items(), gene_colors)):
        # Extract labels and ensure values are numpy arrays
        labels = list(entropy_dict.keys())
        data = [entropy_dict[label] for label in labels]
        data = [array[~np.isnan(array)] for array in data]

        # Create the box plot
        ax = axs[i]
        ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,  # Enables filling boxes with color
            boxprops={"facecolor": color, "alpha": 0.7},
            medianprops={"color": "black", "linewidth": 1},
            flierprops={"marker": "o", "alpha": 0.3, "markersize": 4},  # Outliers
        )
        ax.set_title(gene)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=30, ha="right")  # type: ignore
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(outfile)


def plot_entropy_correlations(level_entropy_dicts, outfile, figsize=(10, 4)):
    fig, axs = plt.subplots(
        1, len(level_entropy_dicts) - 1, figsize=figsize, sharex=True, sharey=True
    )  # type: ignore
    axs: list[Axes] = list(axs)

    levels = list(level_entropy_dicts.keys())

    for i in range(len(level_entropy_dicts) - 1):
        x_level = levels[i + 1]
        y_level = levels[i]

        ax = axs[i]
        ax.scatter(level_entropy_dicts[x_level], level_entropy_dicts[y_level], s=4.0, alpha=0.5)
        ax.set_xlabel(x_level)
        ax.set_ylabel(y_level)

    fig.tight_layout()
    fig.savefig(outfile)


def load_seq_dict(
    fasta_path: Path, constructor: type[skbio.DNA | skbio.Protein]
) -> dict[str, skbio.DNA | skbio.RNA | skbio.Protein]:
    return {
        seq.metadata["id"]: seq
        for seq in skbio.read(fasta_path, format="fasta", constructor=constructor)
    }


def filter_msa(
    msa_index: dict[str, int], msa_matrix: np.ndarray, filter_ids: list[str]
) -> tuple[dict[str, int], np.ndarray]:
    filter_idx = [msa_index[id] for id in filter_ids]
    new_msa_index = {id: i for i, id in enumerate(filter_ids)}
    new_msa_matrix = msa_matrix[filter_idx]

    return new_msa_index, new_msa_matrix


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df = pd.read_csv(cfg.input_metadata, index_col="accession")

    # 18S

    seqs_18s_ref = load_seq_dict(cfg.input_18s_ref_fasta, skbio.DNA)
    seqs_18s_otu = load_seq_dict(cfg.input_18s_otu_fasta, skbio.DNA)
    seqs_18s = seqs_18s_ref | seqs_18s_otu

    msa_18s_index, msa_18s_matrix = create_minimal_msa(seqs_18s_ref)
    msa_18s_length = msa_18s_matrix.shape[1]

    ids_18s_ref = list(seqs_18s_ref.keys())
    ids_18s_otu = list(seqs_18s_otu.keys())

    # 28S

    seqs_28s_ref = load_seq_dict(cfg.input_28s_ref_fasta, skbio.DNA)

    msa_28s_index, msa_28s_matrix = create_minimal_msa(seqs_28s_ref)
    msa_28s_length = msa_28s_matrix.shape[1]

    ids_28s_ref = list(seqs_28s_ref.keys())

    # COI

    prot_seqs_coi = load_seq_dict(cfg.input_coi_ref_and_otu_prot_fasta, skbio.Protein)

    nuc_seqs_coi_ref = load_seq_dict(cfg.input_coi_ref_trimmed_fasta, skbio.DNA)
    nuc_seqs_coi_otu = load_seq_dict(cfg.input_coi_otu_trimmed_fasta, skbio.DNA)
    nuc_seqs_coi = nuc_seqs_coi_ref | nuc_seqs_coi_otu

    prot_seqs_coi_ref = {id: prot_seqs_coi[id] for id in nuc_seqs_coi_ref}
    prot_seqs_coi_otu = {id: prot_seqs_coi[id] for id in nuc_seqs_coi_otu}

    msa_coi_index, msa_coi_matrix = create_minimal_msa(prot_seqs_coi_ref, nuc_seqs=nuc_seqs_coi_ref)
    msa_coi_length = msa_coi_matrix.shape[1]

    ids_coi_ref = list(nuc_seqs_coi_ref.keys())
    ids_coi_otu = list(nuc_seqs_coi_otu.keys())

    # Combine same voucher rows

    df_copy = df.reset_index()

    df_copy["voucher"] = df_copy.voucher.fillna("")

    voucher_trans = str.maketrans({" ": "", "-": "", "_": ""})
    df_copy["voucher_cleaned"] = df_copy.apply(
        lambda row: (
            row.voucher.lower()
            .translate(voucher_trans)
            .removesuffix("18s")
            .removesuffix(row.worms_name.lower().translate(voucher_trans))
        ),
        axis=1,
    )

    no_voucher = df_copy.voucher_cleaned == ""
    df_copy.loc[no_voucher, "voucher_cleaned"] = "NA_" + df_copy.loc[no_voucher, "accession"]

    pivot_df = df_copy.pivot_table(
        index=[
            "trait_genus",
            "worms_class",
            "worms_order",
            "worms_family",
            "worms_genus",
            "worms_name",
            "worms_rank",
            "voucher_cleaned",
        ],
        columns="query_gene",
        values=["accession", "length"],
        aggfunc="first",
    ).reset_index()

    pivot_df["joined_name"] = (
        pivot_df.worms_name
        + " "
        + pivot_df.voucher_cleaned
        + " "
        + pivot_df.accession["18S"].astype(str)
        + " "
        + pivot_df.accession["28S"].astype(str)
        + " "
        + pivot_df.accession["COI"].astype(str)
    )

    pivot_df["joined_length"] = pivot_df.length.fillna(0).sum(axis=1)  # type: ignore

    # has_gene: pd.DataFrame = ~pivot_df.accession.isna()  # type: ignore
    # has_gene.columns = pd.MultiIndex.from_product([["has_gene"], has_gene.columns])

    # pivot_df = pd.concat([pivot_df, has_gene], axis=1)

    # Combine into a single multi gene alignment

    msa_joined_length = msa_18s_length + msa_28s_length + msa_coi_length
    msa_joined_matrix = np.full((len(pivot_df), msa_joined_length), ord("-"), dtype=np.uint8)

    start_col_18s, end_col_18s = 0, msa_18s_length
    start_col_28s, end_col_28s = msa_18s_length, msa_18s_length + msa_28s_length
    start_col_coi, end_col_coi = msa_18s_length + msa_28s_length, msa_joined_length

    print(f"18S: {start_col_18s + 1}-{end_col_18s}")
    print(f"28S: {start_col_28s + 1}-{end_col_28s}")
    print(f"COI: {start_col_coi + 1}-{end_col_coi}")

    for gene, start_col, end_col, msa_index, msa_matrix in (
        ("18S", start_col_18s, end_col_18s, msa_18s_index, msa_18s_matrix),
        ("28S", start_col_28s, end_col_28s, msa_28s_index, msa_28s_matrix),
        ("COI", start_col_coi, end_col_coi, msa_coi_index, msa_coi_matrix),
    ):
        insert_series = pivot_df[("accession", gene)].dropna()
        insert_idx = insert_series.index.to_list()
        insert_ids = insert_series.to_list()

        _, insert_matrix = filter_msa(msa_index, msa_matrix, insert_ids)
        msa_joined_matrix[insert_idx, start_col:end_col] = insert_matrix

    for species_name, species_rows in pivot_df.groupby("worms_name"):
        species_rows = species_rows.sort_values("joined_length", ascending=False)

        species_idx = species_rows.index.to_numpy()
        species_msa_matrix = msa_joined_matrix[species_idx]

        nondegen_mask = np.isin(species_msa_matrix, [ord(c) for c in ["A", "C", "G", "T"]])
        mismatches = species_msa_matrix[:, None, :] != species_msa_matrix[None, :, :]
        shared_nondegen = nondegen_mask[:, None, :] & nondegen_mask[None, :, :]
        actual_mismatches = np.sum(mismatches & shared_nondegen, axis=2)
        comparable_sites = np.sum(shared_nondegen, axis=2)

        p_dist_matrix = actual_mismatches / comparable_sites

        low_distance = np.zeros(len(species_rows), dtype=bool)
        for i in range(len(species_rows)):
            low_distance_idx = np.flatnonzero(p_dist_matrix[i, i + 1 :] < 0.01) + i + 1
            low_distance[low_distance_idx] = True

        pivot_df.loc[species_idx, "cluster_rep"] = ~low_distance

    pivot_df = pivot_df[pivot_df["cluster_rep"]]
    msa_joined_matrix = msa_joined_matrix[pivot_df.index.to_numpy()]
    pivot_df = pivot_df.reset_index(drop=True)

    tax_levels = ["class", "order", "family", "genus"]
    level_up_map = {
        level: level_up for level, level_up in zip(tax_levels, ["global"] + tax_levels[:-1])
    }

    level_entropy = calculate_species_balanced_column_entropies(
        pivot_df, msa_joined_matrix, tax_levels
    )

    gene_global_gainratio = {}
    gene_level_gainratio = {}
    for gene, start_col, end_col, offset, step in (
        ("18S", start_col_18s, end_col_18s, 0, 1),
        ("28S", start_col_28s, end_col_28s, 0, 1),
        ("COI 1st", start_col_coi, end_col_coi, 0, 3),
        ("COI 2nd", start_col_coi, end_col_coi, 1, 3),
        ("COI 3rd", start_col_coi, end_col_coi, 2, 3),
    ):
        gene_global_gainratio[gene] = {
            level: np.clip(
                level_entropy["global"][start_col:end_col][offset::step]
                - level_entropy[level][start_col:end_col][offset::step],
                a_min=0.0,
                a_max=None,
            )
            / level_entropy["global"][start_col:end_col][offset::step]
            for level in tax_levels
        }

        gene_level_gainratio[gene] = {
            level: np.clip(
                level_entropy[level_up_map[level]][start_col:end_col][offset::step]
                - level_entropy[level][start_col:end_col][offset::step],
                a_min=0.0,
                a_max=None,
            )
            / level_entropy[level_up_map[level]][start_col:end_col][offset::step]
            for level in tax_levels
        }

    plot_entropy_boxplots(
        gene_global_gainratio,
        ["coral", "skyblue", "lightgreen", "lightgreen", "lightgreen"],
        outfile="gainratio_global_boxplot.png",
    )

    plot_entropy_boxplots(
        gene_level_gainratio,
        ["coral", "skyblue", "lightgreen", "lightgreen", "lightgreen"],
        outfile="gainratio_level_boxplot.png",
    )

    level_gainratio = {
        level: np.clip(
            level_entropy[level_up_map[level]] - level_entropy[level],
            a_min=0.0,
            a_max=None,
        )
        / level_entropy[level_up_map[level]]
        for level in tax_levels
    }

    for group_name, group_rows in pivot_df.groupby("worms_family", sort=False):
        group_idx = group_rows.index.to_numpy()
        group_msa = msa_joined_matrix[group_idx]

        nondegen_mask = np.isin(group_msa, [ord(c) for c in ["A", "C", "G", "T"]])
        msa_genus_gainratio = nondegen_mask * np.nan_to_num(level_gainratio["genus"])
        overlap_gainratio = nondegen_mask @ msa_genus_gainratio.T

        pivot_df.loc[group_idx, "order_overlap_gainratio"] = np.mean(overlap_gainratio, axis=0)

    pivot_df["genus_rep"] = False
    for (group_rank, group_name), group_rows in pivot_df.groupby(
        ["worms_rank", "worms_name"], sort=False
    ):
        high_mask = (
            group_rows.order_overlap_gainratio >= 0.8 * group_rows.order_overlap_gainratio.max()
        )
        rep_rows = group_rows[high_mask].sort_values("order_overlap_gainratio", ascending=False)
        rep_rows = rep_rows.head(1) if group_rank == "Species" else rep_rows.head(5)

        rep_idx = rep_rows.index.to_numpy()
        pivot_df.loc[rep_idx, "genus_rep"] = True

    pivot_df = pivot_df[pivot_df.genus_rep & (pivot_df.order_overlap_gainratio >= 60)]
    msa_joined_matrix = msa_joined_matrix[pivot_df.index.to_numpy()]
    pivot_df = pivot_df.reset_index(drop=True)

    pivot_df.to_csv(cfg.output_metadata)

    seq_records = [skbio.DNA(row) for row in msa_joined_matrix]

    newick_compat = str.maketrans({" ": "_", "(": "", ")": "", ":": ""})
    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{name.translate(newick_compat)}\n{seq}\n"
            for name, seq in zip(pivot_df.joined_name, seq_records)
        )


if __name__ == "__main__":
    main()
