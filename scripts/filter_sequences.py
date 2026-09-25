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
    input_trait_genera: Path
    output_fasta: Path
    output_outgroup: Path
    output_genus_cov: Path
    no_duplicates: bool
    no_contained: bool
    only_centroids: bool
    otu_coverage: float
    max_ambiguity: float
    min_length: int
    dist_model: str
    model_gamma: float
    max_z_score: float
    k_closest: int


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


def update_genus_coverage(
    df: pd.DataFrame, label: str, old_genus_cov: pd.DataFrame
) -> pd.DataFrame:
    genus_cov = df.groupby("worms_valid_genus").agg({"length": "count"})
    genus_cov.rename(columns={"length": label}, inplace=True)

    return old_genus_cov.join(genus_cov).fillna(0)


def region_coverage(msa_pos, start, end, region_length):
    n_covered = np.sum((msa_pos >= start) & (msa_pos <= end))
    return n_covered / region_length


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    pa_table = pq.read_table(cfg.input_parquet)
    df: pd.DataFrame = pa_table.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
    df.set_index("accession", inplace=True)

    df_ref = df[df.seq_type == "ref"]
    df_out = df[df.seq_type == "outgroup"]
    df_otu = df[df.seq_type == "otu"]

    df_genera = pd.read_excel(cfg.input_trait_genera)
    df_genera.set_index("Genus", inplace=True)

    df_ref = df_ref[df_ref.worms_valid_genus.isin(df_genera.index)]

    genus_cov = update_genus_coverage(df_ref, "Trait genera", df_genera)

    df_ref = df_ref[df_ref.length >= cfg.min_length]

    genus_cov = update_genus_coverage(df_ref, f"Only length >= {cfg.min_length}", genus_cov)

    df_ref = df_ref[df_ref.ambiguity_frac <= cfg.max_ambiguity]

    genus_cov = update_genus_coverage(df_ref, f"Max ambiguity <= {cfg.max_ambiguity}", genus_cov)

    otu_frame_start = df_otu.first_pos.median()
    otu_frame_end = df_otu.last_pos.median()
    otu_length_median = df_otu.length.median()

    df_ref["otu_coverage"] = df_ref.msa_pos.apply(
        lambda pos: region_coverage(pos, otu_frame_start, otu_frame_end, otu_length_median)
    )
    df_ref = df_ref[df_ref.otu_coverage >= cfg.otu_coverage]

    genus_cov = update_genus_coverage(df_ref, f"OTU coverage >= {cfg.otu_coverage}", genus_cov)

    df_ref = df_ref.sort_values(
        ["worms_valid_genus", "clust_id", "otu_coverage", "length", "ambiguity_frac"],
        ascending=[True, True, False, False, True],
    )

    df_ref = df_ref.groupby(["worms_valid_genus", "clust_id"], sort=False, group_keys=False).head(1)

    genus_cov = update_genus_coverage(df_ref, "One per genus x cluster", genus_cov)

    msa = skbio.TabularMSA(create_compressed_alignment(df_ref))
    dist = align_dists(msa, metric=cfg.dist_model, gamma=cfg.model_gamma, shared_by_all=False)
    dist_data = dist.data
    np.fill_diagonal(dist_data, np.nan)

    median_dist = np.nanmedian(dist_data, axis=1)
    z_scores = (median_dist - median_dist.mean()) / median_dist.std()

    print_phylo_z_distribution(z_scores, outlier_threshold=cfg.max_z_score)

    keep_indices = np.flatnonzero(z_scores <= cfg.max_z_score)
    df_ref = df_ref.iloc[keep_indices]
    dist_data = dist_data[np.ix_(keep_indices, keep_indices)]

    genus_cov = update_genus_coverage(df_ref, f"Dist Z-score <= {cfg.max_z_score}", genus_cov)

    genera = df_ref.worms_vald_genus.to_numpy()
    median_genus_dist = np.full(len(genera), np.nan)

    for genus in np.unique(genera):
        genus_idx = np.flatnonzero(genera == genus)

        if len(genus_idx) == 1:
            continue

        genus_dist = dist_data[np.ix_(genus_idx, genus_idx)]

        median_genus_dist[genus_idx] = np.nanmedian(genus_dist, axis=1)

    df_ref["median_genus_dist"] = median_genus_dist

    df_ref = df_ref.sort_values(
        ["worms_valid_genus", "median_genus_dist", "otu_coverage", "length", "ambiguity_frac"],
        ascending=[True, True, False, False, True],
    )

    df_ref = df_ref.groupby("worms_valid_genus", sort=False, group_keys=False).head(3)

    genus_cov = update_genus_coverage(df_ref, "Top 3 median genus dist", genus_cov)

    # dist = dist[np.ix_(keep_indices, keep_indices)]
    # k_closest = np.argpartition(dist, kth=cfg.k_closest + 1, axis=1)[:, 1 : cfg.k_closest + 1]
    # genera = df_ref.worms_valid_genus.to_numpy()

    # _, inverse, counts = np.unique(genera, return_counts=True, return_inverse=True)
    # is_singleton = counts[inverse] == 1

    # closest_genera = genera[k_closest]
    # own_genus = genera[:, None]

    # matches_closest = np.any(closest_genera == own_genus, axis=1) | is_singleton
    # df_ref = df_ref.iloc[matches_closest]

    # genus_cov = update_genus_coverage(df_ref, f"same genus in dist top {cfg.k_closest}", genus_cov)

    df_final = df.loc[df_ref.index.union(df_out.index)]

    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{accession}_{row.worms_valid_genus}\n{seq}\n"
            for (accession, row), seq in zip(
                df_final.iterrows(), create_compressed_alignment(df_final)
            )
        )

    with open(cfg.output_outgroup, "w") as f:
        f.write(
            ",".join(f"{accession}_{row.worms_valid_genus}" for accession, row in df_out.iterrows())
        )

    genus_cov.to_csv(cfg.output_genus_cov)


if __name__ == "__main__":
    main()
