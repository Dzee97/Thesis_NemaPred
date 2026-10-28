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


def calculate_column_statistics(msa_matrix, min_nongap_elements=20, min_nongap_element_count=2):
    nongap_mask = msa_matrix != ord("-")
    col_coverage = np.sum(nongap_mask, axis=0) / len(msa_matrix)

    col_entropies = []

    for col_idx in range(msa_matrix.shape[1]):
        nongap_elements = msa_matrix[nongap_mask[:, col_idx], col_idx]

        if len(nongap_elements) < min_nongap_elements:
            col_entropies.append(0.0)
            continue

        symbols, counts = np.unique(nongap_elements, return_counts=True)

        informative_mask = counts >= min_nongap_element_count
        if len(symbols[informative_mask]) < 2:
            col_entropies.append(0.0)
            continue

        probs = counts / len(nongap_elements)

        col_entropies.append(entropy(probs, base=2))

    return np.array(col_entropies), col_coverage


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


def load_seq_dict(
    fasta_path: Path, constructor: type[skbio.DNA | skbio.RNA | skbio.Protein]
) -> dict[str, skbio.DNA | skbio.RNA | skbio.Protein]:
    return {
        seq.metadata["id"]: seq
        for seq in skbio.read(fasta_path, format="fasta", constructor=constructor)
    }


def get_species_balanced_accessions(
    df: pd.DataFrame, accessions: list[str], groupby="worms_order"
) -> dict[str, list[str]]:
    df = df.loc[accessions]
    group_accessions = {}

    for worms_family, group in df.groupby(groupby):
        non_species_accessions = group[group.worms_rank != "Species"].index.to_list()
        first_species_accessions = (
            group[group.worms_rank == "Species"]
            .groupby("worms_name", sort=False)
            .head(1)
            .index.to_list()
        )
        group_accessions[worms_family] = non_species_accessions + first_species_accessions

    return group_accessions


def get_species_balanced_representatives(df: pd.DataFrame):
    df = df.sort_values(by=[("length", gene) for gene in ("18S", "28S", "COI")], ascending=False)  # type: ignore

    gene_rep_ids = {}
    for gene in ("18S", "28S", "COI"):
        df_gene = df[df.has_gene[gene]]

        non_species_ids = df_gene[df_gene.worms_rank != "Species"].accession[gene].to_list()
        species_rep_ids = (
            df_gene[df_gene.worms_rank == "Species"]
            .groupby("worms_name", sort=False)
            .head(1)
            .accession[gene]
        ).to_list()

        gene_rep_ids[gene] = non_species_ids + species_rep_ids

    return gene_rep_ids


def filter_msa(
    msa_index: dict[str, int], msa_matrix: np.ndarray, filter_ids: list[str]
) -> tuple[dict[str, int], np.ndarray]:
    filter_idx = [msa_index[id] for id in filter_ids]
    new_msa_index = {id: i for i, id in enumerate(filter_ids)}
    new_msa_matrix = msa_matrix[filter_idx]

    return new_msa_index, new_msa_matrix


def calculate_global_and_grouped_column_statistics(msa_index, msa_matrix, group_ids):
    all_ids = [id for group in group_ids.values() for id in group]
    _, global_msa_matrix = filter_msa(msa_index, msa_matrix, all_ids)

    global_entropy, global_coverage = calculate_column_statistics(global_msa_matrix)

    group_index = {}
    group_entropy = np.empty((len(group_ids), len(global_entropy)))
    group_coverage = np.empty((len(group_ids), len(global_coverage)))

    for i, (group, ids) in enumerate(group_ids.items()):
        print(f"Processing {group}, size {len(ids)}")
        group_index[group] = i

        _, order_balanced_msa_18s = filter_msa(msa_index, msa_matrix, ids)
        group_entropy[i], group_coverage[i] = calculate_column_statistics(order_balanced_msa_18s)

    return global_entropy, global_coverage, group_index, group_entropy, group_coverage


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df = pd.read_parquet(cfg.input_parquet, engine="pyarrow", dtype_backend="pyarrow")

    # 18S

    seqs_18s_ref = load_seq_dict(cfg.input_18s_ref_fasta, skbio.RNA)
    seqs_18s_otu = load_seq_dict(cfg.input_18s_otu_fasta, skbio.RNA)
    seqs_18s = seqs_18s_ref | seqs_18s_otu

    msa_18s_index, msa_18s_matrix = create_minimal_msa(seqs_18s)
    msa_18s_length = msa_18s_matrix.shape[1]

    ids_18s_ref = list(seqs_18s_ref.keys())
    ids_18s_otu = list(seqs_18s_otu.keys())

    # 28S

    seqs_28s_ref = load_seq_dict(cfg.input_28s_ref_fasta, skbio.RNA)

    msa_28s_index, msa_28s_matrix = create_minimal_msa(seqs_28s_ref)
    msa_28s_length = msa_28s_matrix.shape[1]

    ids_28s_ref = list(seqs_28s_ref.keys())

    # COI

    prot_seqs_coi = load_seq_dict(cfg.input_coi_ref_and_otu_prot_fasta, skbio.Protein)
    nuc_seqs_coi_ref = load_seq_dict(cfg.input_coi_ref_trimmed_fasta, skbio.DNA)
    nuc_seqs_coi_otu = load_seq_dict(cfg.input_coi_otu_trimmed_fasta, skbio.DNA)
    nuc_seqs_coi = nuc_seqs_coi_ref | nuc_seqs_coi_otu

    msa_coi_index, msa_coi_matrix = create_minimal_msa(prot_seqs_coi, nuc_seqs=nuc_seqs_coi)
    msa_coi_length = msa_coi_matrix.shape[1]

    ids_coi_ref = list(nuc_seqs_coi_ref.keys())
    ids_coi_otu = list(nuc_seqs_coi_otu.keys())

    # Combine same voucher rows

    df_copy = df.reset_index()
    df_copy.loc[df_copy.voucher.isna(), "voucher"] = (
        "NA_" + df_copy.loc[df_copy.voucher.isna(), "accession"]
    )
    pivot_df = (
        df_copy.pivot_table(
            index=[
                "trait_genus",
                "worms_class",
                "worms_order",
                "worms_family",
                "worms_genus",
                "worms_name",
                "worms_rank",
                "ncbi_species",
                "voucher",
            ],
            columns="query_gene",
            values=["accession", "length"],
            aggfunc="first",
        ).reset_index()
        # .sort_values(by=[("length", gene) for gene in ("18S", "28S", "COI")], ascending=False)  # type: ignore
    )

    has_gene: pd.DataFrame = ~pivot_df.accession.isna()  # type: ignore
    has_gene.columns = pd.MultiIndex.from_product([["has_gene"], has_gene.columns])

    pivot_df = pd.concat([pivot_df, has_gene], axis=1)

    # Get species balanced representative accessions for all genes

    gene_reps_ids = get_species_balanced_representatives(pivot_df)

    # Combine into a single multi gene alignment

    msa_joined_length = msa_18s_length + msa_28s_length + msa_coi_length
    msa_joined_matrix = np.full((len(pivot_df), msa_joined_length), ord("-"), dtype=np.uint8)

    joined_reps_entropy = np.zeros(msa_joined_length)
    joined_reps_coverage = np.zeros(msa_joined_length)

    for gene, start_col, end_col, msa_index, msa_matrix in (
        ("18S", 0, msa_18s_length, msa_18s_index, msa_18s_matrix),
        ("28S", msa_18s_length, msa_18s_length + msa_28s_length, msa_28s_index, msa_28s_matrix),
        ("COI", msa_18s_length + msa_28s_length, msa_joined_length, msa_coi_index, msa_coi_matrix),
    ):
        insert_series = pivot_df[("accession", gene)].dropna()
        insert_idx = insert_series.index.to_list()
        insert_ids = insert_series.to_list()

        _, insert_matrix = filter_msa(msa_index, msa_matrix, insert_ids)
        msa_joined_matrix[insert_idx, start_col:end_col] = insert_matrix

        # Calculate column entropies for gene representatives

        _, msa_gene_reps = filter_msa(msa_index, msa_matrix, gene_reps_ids[gene])
        gene_reps_entropy, gene_reps_coverage = calculate_column_statistics(msa_gene_reps)
        joined_reps_entropy[start_col:end_col] = gene_reps_entropy
        joined_reps_coverage[start_col:end_col] = gene_reps_coverage

    nongap_mask = msa_joined_matrix != ord("-")
    msa_joined_matrix_entropy = nongap_mask * joined_reps_entropy
    msa_joined_entropy_overlap = nongap_mask @ msa_joined_matrix_entropy.T

    plot_entropy_overlap_histogram(msa_joined_entropy_overlap, "joined_hist_entropy_overlap.png")
    plot_entropy_overlap_cdf(msa_joined_entropy_overlap, "joined_entropy_overlap.png")

    breakpoint()


if __name__ == "__main__":
    main()
