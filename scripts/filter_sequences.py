from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skbio
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from skbio.alignment import align_dists
from sklearn.metrics import silhouette_samples
from src.common import parse_config


@dataclass
class Config:
    input_parquet: Path
    input_trait_genera: Path
    output_fasta: Path
    output_genus_cov: Path
    min_length: int
    max_ambiguity: float
    otu_coverage: float
    dist_model: str
    model_gamma: float
    max_z_score: float
    min_length_frac: float


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


# def update_genus_coverage(df: pd.DataFrame, label: str, genus_cov: pd.DataFrame) -> pd.DataFrame:
#    new_col = (
#        df.groupby(["worms_order", "worms_family", "worms_genus", "is_target"])
#        .size()
#        .astype("Int64")
#    )
#    genus_cov[label] = new_col
#
#    return genus_cov


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
    df_otu = df[df.seq_type == "otu"]

    df_genera = pd.read_excel(
        cfg.input_trait_genera,
        names=["worms_class", "worms_order", "worms_family", "worms_genus"],
    )
    df_genera["traits_avail"] = True

    # 1. Set genera that are in the trait dataset
    df_ref["traits_avail"] = df_ref.worms_genus.isin(df_genera.worms_genus)

    # 2. Filter out all sequences that dont share an order with the trait genera and are not marine
    df_ref = df_ref[
        (~df_ref.worms_genus.isna())
        & (df_ref.worms_ismarine)
        & (df_ref.worms_order.isin(df_ref[df_ref.traits_avail].worms_order))
    ]

    genus_cov_index_cols = [
        "worms_class",
        "worms_order",
        "worms_family",
        "worms_genus",
        "traits_avail",
    ]
    genus_cov_index = pd.MultiIndex.from_frame(df_genera[genus_cov_index_cols]).union(
        pd.MultiIndex.from_frame(df_ref[genus_cov_index_cols].drop_duplicates())
    )
    genus_cov = pd.DataFrame(index=genus_cov_index)
    genus_cov.sort_index(level="traits_avail", ascending=False, sort_remaining=False, inplace=True)

    genus_cov["Start"] = df_ref.groupby(genus_cov_index_cols).size().astype("Int64")

    # 3. Filter out short sequences
    df_ref = df_ref[df_ref.length >= cfg.min_length]

    genus_cov[f"Length >= {cfg.min_length}"] = (
        df_ref.groupby(genus_cov_index_cols).size().astype("Int64")
    )

    # 4. Filter out low quality sequences
    df_ref = df_ref[df_ref.ambiguity_frac <= cfg.max_ambiguity]

    genus_cov[f"Ambiguity <= {cfg.max_ambiguity}"] = (
        df_ref.groupby(genus_cov_index_cols).size().astype("Int64")
    )

    otu_frame_start = df_otu.first_pos.median()
    otu_frame_end = df_otu.last_pos.median()
    otu_length_median = df_otu.length.median()

    # 5. Filter out sequences that dont span the OTU frame
    df_ref["otu_coverage"] = df_ref.msa_pos.apply(
        lambda pos: region_coverage(pos, otu_frame_start, otu_frame_end, otu_length_median)
    )
    df_ref = df_ref[df_ref.otu_coverage >= cfg.otu_coverage]

    genus_cov[f"OTU coverage >= {cfg.otu_coverage}"] = (
        df_ref.groupby(genus_cov_index_cols).size().astype("Int64")
    )

    # 6. Group by (genus, cluster) and keep only the longest sequence
    df_ref = df_ref.sort_values(
        ["worms_genus", "clust_id", "length", "otu_coverage", "ambiguity_frac"],
        ascending=[True, True, False, False, True],
    )
    df_ref = df_ref.groupby(["worms_genus", "clust_id"], sort=False, group_keys=False).head(1)

    genus_cov["Genus x global cluster"] = (
        df_ref.groupby(genus_cov_index_cols).size().astype("Int64")
    )

    msa = skbio.TabularMSA(create_compressed_alignment(df_ref), index=df_ref.index)
    dist = align_dists(msa, metric=cfg.dist_model, gamma=cfg.model_gamma, shared_by_all=False)
    dist.data[dist.data < 0.0] = 0.0

    dist_data = dist.data.copy()
    np.fill_diagonal(dist_data, np.nan)

    median_dist = np.nanmedian(dist_data, axis=1)
    z_scores = (median_dist - median_dist.mean()) / median_dist.std()

    print_phylo_z_distribution(z_scores, outlier_threshold=cfg.max_z_score)

    # 7. Filter out sequences that are extermely divergent to the other sequences
    keep_indices = np.flatnonzero(z_scores <= cfg.max_z_score)
    df_ref = df_ref.iloc[keep_indices]
    dist = dist.filter(list(df_ref.index))

    genus_cov[f"TN93 Z-score <= {cfg.max_z_score}"] = (
        df_ref.groupby(genus_cov_index_cols).size().astype("Int64")
    )

    # 8. Perform hierarchical clustering on every genus
    genera = df_ref.worms_genus.to_numpy()
    genera_clusters = np.arange(len(genera), dtype=np.int16)
    genera_silhouette_coefs = np.full(len(genera), np.nan)
    genera_linkages = {}

    for genus in np.unique(genera):
        genus_idx = np.flatnonzero(genera == genus)

        if len(genus_idx) < 4:
            continue

        genus_dist = dist.filter(list(df_ref.index[genus_idx]))
        genus_linkage = linkage(genus_dist.condensed_form(), method="average")
        genera_linkages[genus] = genus_linkage

        best_score = -1

        for k in range(3, 4):
            labels = fcluster(genus_linkage, k, criterion="maxclust")
            coefs = silhouette_samples(genus_dist.data, labels, metric="precomputed")
            score = np.asarray(coefs).mean()

            if score > best_score:
                best_score = score
                best_coefs = coefs
                best_k = k

        optimal_labels = fcluster(genus_linkage, best_k, criterion="maxclust")

        print(f"Genus: {genus}, Size: {len(genus_idx)}, Optimal clusters: {best_k}")

        genera_clusters[genus_idx] = optimal_labels
        genera_silhouette_coefs[genus_idx] = best_coefs

    df_ref["genus_cluster"] = genera_clusters
    df_ref["genus_silhouette_coefs"] = genera_silhouette_coefs

    df_ref["genus_long_enough"] = df_ref.length >= (
        df_ref.groupby(["worms_genus", "genus_cluster"]).length.transform("max")
        * cfg.min_length_frac
    )

    df_ref = df_ref.sort_values(
        [
            "worms_genus",
            "genus_cluster",
            "genus_long_enough",
            "genus_silhouette_coefs",
            "length",
            "otu_coverage",
            "ambiguity_frac",
        ],
        ascending=[True, True, False, False, False, False, True],
    )

    df_ref = df_ref.groupby(["worms_genus", "genus_cluster"], sort=False, group_keys=False).head(1)

    genus_cov["Genus cluster reps"] = df_ref.groupby(genus_cov_index_cols).size().astype("Int64")

    genus_cov.to_csv(cfg.output_genus_cov)

    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{accession}_{row.worms_genus}\n{seq}\n"
            for (accession, row), seq in zip(df_ref.iterrows(), create_compressed_alignment(df_ref))
        )


if __name__ == "__main__":
    main()
