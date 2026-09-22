from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skbio
from skbio.alignment import align_dists
from src.common import parse_config
from tqdm import tqdm


@dataclass
class Config:
    input_parquet: Path
    output_fasta: Path
    no_duplicates: bool
    no_contained: bool
    only_cluster_centroids: bool
    min_length: int
    dist_model: str
    model_gamma: float
    max_z_score: float


def print_phylo_z_distribution(z_arr, outlier_threshold=2.5):
    total = len(z_arr)

    # 1. Categorize into customized evolutionary distance bins
    close_conserved = np.sum(z_arr < 0)  # Closer than average
    mod_divergent = np.sum((z_arr >= 0) & (z_arr < 1.0))
    highly_divergent = np.sum((z_arr >= 1.0) & (z_arr < outlier_threshold))
    potential_outliers = np.sum(z_arr >= outlier_threshold)  # Long divergent tail

    # Custom row formatting function
    def format_row(label, count, description):
        pct = (count / total) * 100
        bar = "█" * int(pct / 2)  # 1 block per 2%
        return f"{label:<25} | {count:>5} ({pct:>5.1f}%) | {description:<30} | {bar}"

    # 2. Print Distribution Report
    print(f"\n{' PHYLOGENETIC SEQUENCE OUTLIER REPORT ':^85}")
    print("=" * 85)
    print(
        f"{'Z-Score (Distance) Band':<25} | {'Count / Pct':<12} | {'Biological Context':<30} | Visual Density"
    )
    print("-" * 85)
    print(format_row("Z < 0 (Negative)", close_conserved, "Highly Conserved / Similar Sequences"))
    print(format_row("0 <= Z < 1.0", mod_divergent, "Standard Evolutionary Variance"))
    print(
        format_row(
            "1.0 <= Z < " + str(outlier_threshold), highly_divergent, "Divergent (Keep in tree)"
        )
    )
    print(
        format_row(
            f"Z >= {outlier_threshold} (Positive)",
            potential_outliers,
            "CRITICAL: Highly Divergent/Outliers",
        )
    )
    print("=" * 85)

    # 3. Basic Metrics
    print(f"Total Sequences Analyzed: {total}")
    print(
        f"Minimum Z-score:          {np.min(z_arr):.2f} (Bounded by absolute sequence identity limits)"
    )
    print(f"Maximum Z-score:          {np.max(z_arr):.2f}")

    # 4. Outlier Recommendations
    outlier_indices = np.where(z_arr >= outlier_threshold)[0]

    print("\n" + "-" * 85)
    if len(outlier_indices) > 0:
        print(
            f"❌ ACTION REQUIRED: Found {len(outlier_indices)} sequence(s) exceeding Z = {outlier_threshold}."
        )
        print(
            "   These sequences have an abnormally high average distance to the rest of the alignment."
        )
        print("   Consider removing them to prevent long-branch attraction artifacts in your tree.")
        print(f"   Outlier Array Indices: {outlier_indices.tolist()}")
        print(f"   Outlier Z-scores:      {np.round(z_arr[outlier_indices], 2).tolist()}")
    else:
        print(
            f"✅ CLEAN SET: No sequences exceeded the positive outlier threshold of Z = {outlier_threshold}."
        )
        print("   Your alignment distances are stable enough for phylogenetic reconstruction.")
    print("-" * 85)


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    pa_table = pq.read_table(cfg.input_parquet)
    df: pd.DataFrame = pa_table.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
    df.set_index("accession", inplace=True)

    df = df[
        (df["is_outgroup"] == True)
        | ((df["length"] >= cfg.min_length) & (df["clust_centroid"] == cfg.only_cluster_centroids))
    ]

    positions_array = pa.array(df.msa_pos)
    combined_positions = positions_array.flatten()
    active_positions = combined_positions.unique().sort().to_numpy()

    aligned_sequences: list[skbio.RNA] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating compressed filtered alignment"):
        aligned_bytes = np.full(len(active_positions), ord("-"), dtype=np.uint8)
        insert_indices = np.searchsorted(active_positions, row.msa_pos)
        aligned_bytes[insert_indices] = row.rna_seq
        aligned_sequences.append(skbio.RNA(aligned_bytes))

    msa = skbio.TabularMSA(aligned_sequences)
    dist = align_dists(msa, metric=cfg.dist_model, gamma=cfg.model_gamma, shared_by_all=False)

    mean_dist = np.nanmedian(dist.data, axis=0)
    z_scores = (mean_dist - mean_dist.mean()) / mean_dist.std()

    print_phylo_z_distribution(z_scores, outlier_threshold=cfg.max_z_score)

    keep_indices = np.flatnonzero((z_scores <= cfg.max_z_score) | df.is_outgroup.to_numpy())

    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{accession}_{row.worms_valid_genus}\n{seq}\n"
            for (accession, row), seq in zip(
                df.iloc[keep_indices].iterrows(), msa.iloc[keep_indices]
            )
        )


if __name__ == "__main__":
    main()
