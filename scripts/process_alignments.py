import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import skbio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import entropy
from src.common import parse_config
from tqdm import tqdm


@dataclass
class Config:
    input_parquet: Path
    input_18s_ref_fasta: Path
    input_18s_otu_fasta: Path
    input_28s_ref_fasta: Path
    input_coi_ref_and_otu_prot_fasta: Path
    input_coi_ref_trimmed_fasta: Path
    input_coi_otu_trimmed_fasta: Path
    output_parquet: Path


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

    if nuc_seqs is None:
        msa_index = {}
        msa_matrix = np.full((len(seq_dict), len(unique_positions)), ord("-"), dtype=np.uint8)

        for i, (accession, seq) in enumerate(seq_dict.items()):
            msa_index[accession] = i

            non_gap_mask = ~seq.gaps()
            new_positions = new_position_map[non_gap_mask]

            ungapped_seq = seq.degap().values.view(np.uint8)
            msa_matrix[i, new_positions] = ungapped_seq

        return msa_index, msa_matrix

    else:
        msa_index = {}
        msa_matrix = np.full((len(seq_dict), len(unique_positions) * 3), ord("-"), dtype=np.uint8)

        for i, (accession, prot_seq) in enumerate(seq_dict.items()):
            msa_index[accession] = i

            non_gap_mask = ~prot_seq.gaps()
            new_positions = new_position_map[non_gap_mask]

            new_positions = new_positions * 3
            new_positions = np.sort(
                np.concatenate((new_positions, new_positions + 1, new_positions + 2))
            )

            nuc_seq = nuc_seqs[accession]
            ungapped_seq = nuc_seq.degap().values.view(np.uint8)
            msa_matrix[i, new_positions] = ungapped_seq

        return msa_index, msa_matrix


def calculate_column_entropies(msa_matrix):
    valid_mask = msa_matrix != ord("-")
    entropies = []
    coverages = []

    for col_idx in range(msa_matrix.shape[1]):
        valid_elements = msa_matrix[valid_mask[:, col_idx], col_idx]
        coverage = len(valid_elements) / len(msa_matrix)
        coverages.append(coverage)

        if coverage < 0.02 or len(valid_elements) < 20:
            entropies.append(0.0)
            continue

        symbols, counts = np.unique(valid_elements, return_counts=True)

        informative_mask = counts >= 2
        if len(symbols[informative_mask]) < 2:
            entropies.append(0.0)
            continue

        probs = counts / len(valid_elements)

        entropies.append(entropy(probs, base=2))

    return np.array(entropies), np.array(coverages)


def plot_entropies(ref_entropies, ref_coverage, outfile, otu_entropies=None, otu_coverage=None):
    """
    Plots column entropies in sequential order and as a distribution.
    """

    x = np.arange(len(ref_entropies))

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 1. Ordered Line Plot
    ax1.scatter(
        x[ref_entropies > 0],
        ref_entropies[ref_entropies > 0],
        c="steelblue",
        alpha=0.8,
        s=6.0,
        label="NCBI Refs",
    )
    if otu_entropies is not None:
        ax1.scatter(
            x[otu_entropies > 0],
            otu_entropies[otu_entropies > 0],
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
        x[ref_coverage > 0],
        ref_coverage[ref_coverage > 0],
        c="steelblue",
        alpha=0.8,
        s=6.0,
        label="NCBI Refs",
    )
    if otu_coverage is not None:
        ax2.scatter(
            x[otu_coverage > 0],
            otu_coverage[otu_coverage > 0],
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


def load_seq_dict(
    fasta_path: Path, constructor: type[skbio.DNA | skbio.RNA | skbio.Protein]
) -> dict[str, skbio.DNA | skbio.RNA | skbio.Protein]:
    return {
        seq.metadata["id"]: seq
        for seq in skbio.read(fasta_path, format="fasta", constructor=constructor)
    }


def get_species_balanced_accessions(df: pd.DataFrame, accessions: list[str]) -> list[str]:
    df = df.loc[accessions]
    non_species_accessions = df[df.worms_rank != "Species"].index.to_list()
    first_species_accessions = (
        df[df.worms_rank == "Species"].groupby("worms_name", sort=False).head(1).index.to_list()
    )
    return non_species_accessions + first_species_accessions


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

    df = pd.read_parquet(cfg.input_parquet, engine="pyarrow", dtype_backend="pyarrow")

    # 18S

    seqs_18s_ref = load_seq_dict(cfg.input_18s_ref_fasta, skbio.RNA)
    seqs_18s_otu = load_seq_dict(cfg.input_18s_otu_fasta, skbio.RNA)
    seqs_18s = seqs_18s_ref | seqs_18s_otu

    ids_18s_ref = list(seqs_18s_ref.keys())
    ids_18s_ref_bal = get_species_balanced_accessions(df, ids_18s_ref)
    ids_18s_otu = list(seqs_18s_otu.keys())

    msa_18s_index, msa_18s_matrix = create_minimal_msa(seqs_18s)
    msa_18s_ref_index, msa_18s_ref_matrix = filter_msa(msa_18s_index, msa_18s_matrix, ids_18s_ref)
    _, msa_18s_ref_bal_matrix = filter_msa(msa_18s_index, msa_18s_matrix, ids_18s_ref_bal)
    _, msa_18s_otu_matrix = filter_msa(msa_18s_index, msa_18s_matrix, ids_18s_otu)

    msa_18s_ref_entropy, msa_18s_ref_coverage = calculate_column_entropies(msa_18s_ref_bal_matrix)
    msa_18s_otu_entropy, msa_18s_otu_coverage = calculate_column_entropies(msa_18s_otu_matrix)

    plot_entropies(
        ref_entropies=msa_18s_ref_entropy,
        ref_coverage=msa_18s_ref_coverage,
        otu_entropies=msa_18s_otu_entropy,
        otu_coverage=msa_18s_otu_coverage,
        outfile="18s_entropy.png",
    )

    valid_mask = msa_18s_ref_matrix != ord("-")
    msa_18s_ref_matrix_entropy = valid_mask * msa_18s_ref_entropy
    msa_18s_ref_entropy_overlap = valid_mask @ msa_18s_ref_matrix_entropy.T

    plot_entropy_overlap_cdf(msa_18s_ref_entropy_overlap, "18s_entropy_overlap.png")

    # 28S

    seqs_28s_ref = load_seq_dict(cfg.input_28s_ref_fasta, skbio.RNA)

    ids_28s_ref = list(seqs_28s_ref.keys())
    ids_28s_ref_bal = get_species_balanced_accessions(df, ids_28s_ref)

    msa_28s_ref_index, msa_28s_ref_matrix = create_minimal_msa(seqs_28s_ref)
    _, msa_28s_ref_bal_matrix = filter_msa(msa_28s_ref_index, msa_28s_ref_matrix, ids_28s_ref_bal)

    msa_28s_ref_entropy, msa_28s_ref_coverage = calculate_column_entropies(msa_28s_ref_bal_matrix)

    plot_entropies(
        ref_entropies=msa_28s_ref_entropy,
        ref_coverage=msa_28s_ref_coverage,
        outfile="28s_entropy.png",
    )

    valid_mask = msa_28s_ref_matrix != ord("-")
    msa_28s_ref_matrix_entropy = valid_mask * msa_28s_ref_entropy
    msa_28s_ref_entropy_overlap = valid_mask @ msa_28s_ref_matrix_entropy.T

    plot_entropy_overlap_cdf(msa_28s_ref_entropy_overlap, "28s_entropy_overlap.png")

    # COI

    prot_seqs_coi = load_seq_dict(cfg.input_coi_ref_and_otu_prot_fasta, skbio.Protein)
    nuc_seqs_coi_ref = load_seq_dict(cfg.input_coi_ref_trimmed_fasta, skbio.DNA)
    nuc_seqs_coi_otu = load_seq_dict(cfg.input_coi_otu_trimmed_fasta, skbio.DNA)
    nuc_seqs_coi = nuc_seqs_coi_ref | nuc_seqs_coi_otu

    ids_coi_ref = list(nuc_seqs_coi_ref.keys())
    ids_coi_ref_bal = get_species_balanced_accessions(df, ids_coi_ref)
    ids_coi_otu = list(nuc_seqs_coi_otu.keys())

    msa_coi_index, msa_coi_matrix = create_minimal_msa(prot_seqs_coi, nuc_seqs=nuc_seqs_coi)
    msa_coi_ref_index, msa_coi_ref_matrix = filter_msa(msa_coi_index, msa_coi_matrix, ids_coi_ref)
    _, msa_coi_ref_bal_matrix = filter_msa(msa_coi_index, msa_coi_matrix, ids_coi_ref_bal)
    _, msa_coi_otu_matrix = filter_msa(msa_coi_index, msa_coi_matrix, ids_coi_otu)

    msa_coi_ref_entropy, msa_coi_ref_coverage = calculate_column_entropies(msa_coi_ref_bal_matrix)
    msa_coi_otu_entropy, msa_coi_otu_coverage = calculate_column_entropies(msa_coi_otu_matrix)

    plot_entropies(
        ref_entropies=msa_coi_ref_entropy,
        ref_coverage=msa_coi_ref_coverage,
        otu_entropies=msa_coi_otu_entropy,
        otu_coverage=msa_coi_otu_coverage,
        outfile="coi_entropy.png",
    )

    valid_mask = msa_coi_ref_matrix != ord("-")
    msa_coi_matrix_entropy = valid_mask * msa_coi_ref_entropy
    msa_coi_entropy_overlap = valid_mask @ msa_coi_matrix_entropy.T

    plot_entropy_overlap_cdf(msa_coi_entropy_overlap, "coi_entropy_overlap.png")

    # Combine same voucher entropies

    df_copy = df.reset_index()
    df_copy.loc[df_copy.voucher.isna(), "voucher"] = (
        "NA_" + df_copy.loc[df_copy.voucher.isna(), "accession"]
    )
    pivot_df = df_copy.pivot_table(
        index=["ncbi_species", "voucher"], columns="query_gene", values="accession", aggfunc="first"
    ).reset_index()

    pivot_df["joined_accession"] = (
        pivot_df["18S"].astype(str)
        + "_"
        + pivot_df["28S"].astype(str)
        + "_"
        + pivot_df["COI"].astype(str)
    )
    # pivot_df.set_index("joined_accession", inplace=True)

    msa_18s_length = msa_18s_ref_matrix.shape[1]
    msa_18s_joined_matrix = np.full((len(pivot_df), msa_18s_length), ord("-"), dtype=np.uint8)

    insert_idx_18s = pivot_df["18S"].dropna().index.to_list()
    insert_ids_18s = pivot_df["18S"].dropna().to_list()

    _, insert_18s_matrix = filter_msa(msa_18s_ref_index, msa_18s_ref_matrix, insert_ids_18s)
    msa_18s_joined_matrix[insert_idx_18s] = insert_18s_matrix

    msa_28s_length = msa_28s_ref_matrix.shape[1]
    msa_28s_joined_matrix = np.full((len(pivot_df), msa_28s_length), ord("-"), dtype=np.uint8)

    insert_idx_28s = pivot_df["28S"].dropna().index.to_list()
    insert_ids_28s = pivot_df["28S"].dropna().to_list()

    _, insert_28s_matrix = filter_msa(msa_28s_ref_index, msa_28s_ref_matrix, insert_ids_28s)
    msa_28s_joined_matrix[insert_idx_28s] = insert_28s_matrix

    msa_coi_length = msa_coi_ref_matrix.shape[1]
    msa_coi_joined_matrix = np.full((len(pivot_df), msa_coi_length), ord("-"), dtype=np.uint8)

    insert_idx_coi = pivot_df["COI"].dropna().index.to_list()
    insert_ids_coi = pivot_df["COI"].dropna().to_list()

    _, insert_coi_matrix = filter_msa(msa_coi_ref_index, msa_coi_ref_matrix, insert_ids_coi)
    msa_coi_joined_matrix[insert_idx_coi] = insert_coi_matrix

    msa_joined_matrix = np.concatenate(
        (msa_18s_joined_matrix, msa_28s_joined_matrix, msa_coi_joined_matrix), axis=1
    )

    valid_mask = msa_joined_matrix != ord("-")
    joined_entropy = np.concatenate((msa_18s_ref_entropy, msa_28s_ref_entropy, msa_coi_ref_entropy))
    msa_joined_matrix_entropy = valid_mask * joined_entropy
    msa_joined_entropy_overlap = valid_mask @ msa_joined_matrix_entropy.T

    plot_entropy_overlap_cdf(msa_joined_entropy_overlap, "joined_entropy_overlap.png")

    breakpoint()


if __name__ == "__main__":
    main()
