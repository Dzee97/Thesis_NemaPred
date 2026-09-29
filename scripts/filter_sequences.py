from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skbio
from skbio.alignment import align_dists
from src.common import parse_config


@dataclass
class Config:
    input_parquet: Path
    input_trait_genera: Path
    output_fasta: Path
    output_outgroup: Path
    output_genus_cov: Path
    min_length: int
    max_ambiguity: float
    otu_coverage: float
    dist_model: str
    model_gamma: float
    max_z_score: float
    eps_quantile: float
    max_centrality: float
    keep_longest: int


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


def create_compressed_alignment(df: pd.DataFrame) -> list[skbio.RNA]:
    positions_array = pa.array(df.msa_pos)
    combined_positions = positions_array.flatten()
    active_positions = combined_positions.unique().sort().to_numpy()

    aligned_sequences: list[skbio.RNA] = []
    for _, row in df.iterrows():
        aligned_bytes = np.full(len(active_positions), ord("-"), dtype=np.uint8)
        insert_indices = np.searchsorted(active_positions, row.msa_pos)
        aligned_bytes[insert_indices] = row.rna_seq
        aligned_sequences.append(skbio.RNA(aligned_bytes))

    return aligned_sequences


def update_genus_coverage(df: pd.DataFrame, label: str, genus_cov: pd.DataFrame) -> pd.DataFrame:
    new_col = (
        df.groupby(["worms_order", "worms_family", "worms_genus", "is_target"])
        .size()
        .astype("Int64")
    )
    genus_cov[label] = new_col

    return genus_cov


def region_coverage(msa_pos, start, end, region_length):
    n_covered = np.sum((msa_pos >= start) & (msa_pos <= end))
    return n_covered / region_length


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    pa_table = pq.read_table(cfg.input_parquet)
    df: pd.DataFrame = pa_table.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
    df.set_index("accession", inplace=True)

    df_ref = df[df.seq_type == "ref"].copy()
    df_out = df[df.seq_type == "outgroup"]
    df_otu = df[df.seq_type == "otu"]

    df_genera = pd.read_excel(cfg.input_trait_genera)
    df_genera.set_index("Genus", inplace=True)

    # 1. Set genera that are in the trait dataset
    df_ref["is_target"] = df_ref.worms_genus.isin(df_genera.index)

    # 2. Filter out all sequences that dont share an order with the trait genera and are not marine
    df_ref = df_ref[
        (~df_ref.worms_genus.isna())
        & (df_ref.worms_ismarine)
        & (df_ref.worms_order.isin(df_ref[df_ref.is_target].worms_order))
    ]

    genus_cov = (
        df_ref.groupby(["worms_order", "worms_family", "worms_genus", "is_target"])
        .size()
        .to_frame(name="Start")
    )

    # 3. Filter out short sequences
    df_ref = df_ref[df_ref.length >= cfg.min_length]

    genus_cov = update_genus_coverage(df_ref, f"Length >= {cfg.min_length}", genus_cov)

    # 4. Filter out low quality sequences
    df_ref = df_ref[df_ref.ambiguity_frac <= cfg.max_ambiguity]

    genus_cov = update_genus_coverage(df_ref, f"Ambiguity <= {cfg.max_ambiguity}", genus_cov)

    otu_frame_start = df_otu.first_pos.median()
    otu_frame_end = df_otu.last_pos.median()
    otu_length_median = df_otu.length.median()

    # 5. Filter out sequences that dont span the OTU frame
    df_ref["otu_coverage"] = df_ref.msa_pos.apply(
        lambda pos: region_coverage(pos, otu_frame_start, otu_frame_end, otu_length_median)
    )
    df_ref = df_ref[df_ref.otu_coverage >= cfg.otu_coverage]

    genus_cov = update_genus_coverage(df_ref, f"OTU coverage >= {cfg.otu_coverage}", genus_cov)

    # 6. Group by (genus, cluster) and keep only the longest sequence
    df_ref = df_ref.sort_values(
        ["worms_genus", "clust_id", "length", "otu_coverage", "ambiguity_frac"],
        ascending=[True, True, False, False, True],
    )
    df_ref = df_ref.groupby(["worms_genus", "clust_id"], sort=False, group_keys=False).head(1)

    genus_cov = update_genus_coverage(df_ref, "Genus x Cluster", genus_cov)

    msa = skbio.TabularMSA(create_compressed_alignment(df_ref))
    dist = align_dists(msa, metric=cfg.dist_model, gamma=cfg.model_gamma, shared_by_all=False)
    dist_data = dist.data
    np.fill_diagonal(dist_data, np.nan)

    median_dist = np.nanmedian(dist_data, axis=1)
    z_scores = (median_dist - median_dist.mean()) / median_dist.std()

    print_phylo_z_distribution(z_scores, outlier_threshold=cfg.max_z_score)

    # 7. Filter out sequences that are extermely divergent to the other sequences
    keep_indices = np.flatnonzero(z_scores <= cfg.max_z_score)
    df_ref = df_ref.iloc[keep_indices]
    dist_data = dist_data[np.ix_(keep_indices, keep_indices)]

    genus_cov = update_genus_coverage(df_ref, f"Dist Z-score <= {cfg.max_z_score}", genus_cov)

    genera = df_ref.worms_genus.to_numpy()
    median_genus_dist = np.full(len(genera), np.nan)
    best_genus_dist = np.full(len(genera), np.nan)

    for genus in np.unique(genera):
        genus_idx = np.flatnonzero(genera == genus)

        if len(genus_idx) == 1:
            continue

        genus_dist = dist_data[np.ix_(genus_idx, genus_idx)]
        median_dist = np.nanmedian(genus_dist, axis=1)

        median_genus_dist[genus_idx] = median_dist
        best_genus_dist[genus_idx] = np.min(median_dist)

    _, genus_first_idx = np.unique(genera, return_index=True)
    eps_values = best_genus_dist[genus_first_idx]
    eps = np.nanquantile(eps_values, cfg.eps_quantile)

    df_ref["central"] = np.isnan(median_genus_dist) | (
        median_genus_dist <= best_genus_dist + cfg.max_centrality * np.maximum(best_genus_dist, eps)
    )

    # 8. Filter out sequences that are more than 50% the lowest centrality score within each genus
    df_ref = df_ref[df_ref.central]

    genus_cov = update_genus_coverage(
        df_ref, f"Central <= {cfg.max_centrality} w/i genus", genus_cov
    )

    df_ref = df_ref.sort_values(
        ["worms_genus", "length", "otu_coverage", "ambiguity_frac"],
        ascending=[True, False, False, True],
    )

    # 9. Keep up to 3 longest sequences per genus
    df_ref = df_ref.groupby("worms_genus", sort=False, group_keys=False).head(cfg.keep_longest)

    genus_cov = update_genus_coverage(df_ref, f"Top {cfg.keep_longest} length", genus_cov)

    df_final = df.loc[df_ref.index.union(df_out.index)]

    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{accession}_{row.worms_genus}\n{seq}\n"
            for (accession, row), seq in zip(
                df_final.iterrows(), create_compressed_alignment(df_final)
            )
        )

    with open(cfg.output_outgroup, "w") as f:
        f.write(",".join(f"{accession}_{row.worms_genus}" for accession, row in df_out.iterrows()))

    genus_cov.to_csv(cfg.output_genus_cov)


if __name__ == "__main__":
    main()
