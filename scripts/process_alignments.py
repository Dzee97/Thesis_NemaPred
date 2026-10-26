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
from scipy.stats import entropy
from src.common import parse_config
from tqdm import tqdm


@dataclass
class Config:
    input_parquet: Path
    input_18s_ref_fasta: Path
    input_18s_otu_fasta: Path
    input_28s_ref_fasta: Path
    input_coi_ref_and_otu_fasta: Path
    output_parquet: Path


def create_minimal_msa(seq_list):
    alignment_length = -1
    unique_positions = set()

    for seq in seq_list:
        if alignment_length < 0:
            alignment_length = len(seq)

        non_gap_mask = ~seq.gaps()
        positions = np.flatnonzero(non_gap_mask).astype(np.uint16)
        unique_positions |= set(positions)

    unique_positions = np.sort(list(unique_positions))
    new_position_map = np.full(alignment_length, -1, dtype=np.int32)
    new_position_map[unique_positions] = np.arange(len(unique_positions))

    msa_index = {}
    msa_matrix = np.full((len(seq_list), len(unique_positions)), ord("-"), dtype=np.uint8)

    for i, seq in enumerate(seq_list):
        msa_index[seq.metadata["id"]] = i

        non_gap_mask = ~seq.gaps()
        ungapped_seq = seq.degap().values.view(np.uint8)
        new_positions = new_position_map[non_gap_mask]
        msa_matrix[i, new_positions] = ungapped_seq

    return msa_index, msa_matrix


def calculate_column_entropies(msa_matrix):
    valid_mask = msa_matrix != ord("-")
    entropies = []

    for col_idx in range(msa_matrix.shape[1]):
        valid_elements = msa_matrix[valid_mask[:, col_idx], col_idx]

        if len(valid_elements) < 10:
            entropies.append(0.0)
            continue

        symbols, counts = np.unique(valid_elements, return_counts=True)

        informative_mask = counts >= 2
        if len(symbols[informative_mask]) < 2:
            entropies.append(0.0)
            continue

        probs = counts / len(valid_elements)

        entropies.append(entropy(probs, base=2))

    return np.array(entropies)


def plot_entropies(entropies, outfile):
    """
    Plots column entropies in sequential order and as a distribution.
    """

    pos_mask = entropies > 0
    x = np.arange(len(entropies))[pos_mask]

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 1. Ordered Line Plot
    ax1.scatter(x, entropies[pos_mask], c=entropies[pos_mask], cmap="RdYlGn", alpha=0.8, s=6.0)
    ax1.set_title("Column Entropies (Alignment Order)")
    ax1.set_xlabel("Alignment Columns")
    ax1.set_ylabel("Shannon Entropy (bits)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Distribution Histogram
    # Automatically determine bins using 'auto', or use a fixed integer like 15
    ax2.hist(entropies[pos_mask], bins=30, color="#2ca02c", edgecolor="black", alpha=0.8)
    ax2.set_title("Distribution of Column Entropies")
    ax2.set_xlabel("Shannon Entropy (bits)")
    ax2.set_ylabel("Frequency (Number of Columns)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Clean up layout and display
    plt.tight_layout()
    plt.savefig(outfile)


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df = pd.read_parquet(cfg.input_parquet, engine="pyarrow", dtype_backend="pyarrow")

    seqs_18s_ref: list[skbio.RNA] = [
        seq for seq in skbio.read(cfg.input_18s_ref_fasta, format="fasta", constructor=skbio.RNA)
    ]
    seqs_18s_otu: list[skbio.RNA] = [
        seq for seq in skbio.read(cfg.input_18s_otu_fasta, format="fasta", constructor=skbio.RNA)
    ]
    seqs_18s = seqs_18s_ref + seqs_18s_otu

    msa_18s_index, msa_18s_matrix = create_minimal_msa(seqs_18s)
    msa_18s_entropy = calculate_column_entropies(msa_18s_matrix)
    plot_entropies(msa_18s_entropy, "18s_entropy.png")

    seqs_28s: list[skbio.RNA] = [
        seq for seq in skbio.read(cfg.input_28s_ref_fasta, format="fasta", constructor=skbio.RNA)
    ]

    msa_28s_index, msa_28s_matrix = create_minimal_msa(seqs_28s)
    msa_28s_entropy = calculate_column_entropies(msa_28s_matrix)
    plot_entropies(msa_28s_entropy, "28s_entropy.png")

    seqs_coi: list[skbio.Protein] = [
        seq
        for seq in skbio.read(
            cfg.input_coi_ref_and_otu_fasta, format="fasta", constructor=skbio.Protein
        )
    ]

    msa_coi_index, msa_coi_matrix = create_minimal_msa(seqs_coi)
    msa_coi_entropy = calculate_column_entropies(msa_coi_matrix)
    plot_entropies(msa_coi_entropy, "coi_entropy.png")

    breakpoint()


if __name__ == "__main__":
    main()
