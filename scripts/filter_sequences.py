from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
    input_trait_data: Path
    output_fasta: Path
    output_genus_cov: Path
    min_length: int
    max_ambiguity: float
    otu_coverage: float
    dist_model: str
    model_gamma: float
    max_z_score: float
    worms_name_cluster_dist: float
    worms_name_length_frac: float


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


def plot_name_tree(linkage, name, labels, threshold, save_path):
    plt.figure(figsize=(8, 5))
    plt.title(f"TN93 Clustering Dendrogram: {name}", fontsize=14)

    dendrogram(
        linkage,
        labels=labels,
        orientation="right",
        color_threshold=0,
        above_threshold_color="skyblue",
    )

    plt.axvline(x=threshold, color="r", linestyle="--", linewidth="2", label="Cutt-off")
    plt.xlim((0, threshold * 2))
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    pa_table = pq.read_table(cfg.input_parquet)
    df: pd.DataFrame = pa_table.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
    df.set_index("accession", inplace=True)

    df_ref = df[df.seq_type == "ref"].copy()

    taxonomy_cols = ["worms_class", "worms_order", "worms_family", "worms_genus"]

    df_trait_genera = pd.read_csv(
        cfg.input_trait_data,
        usecols=taxonomy_cols,
    )

    # 1. Filter out all sequences that dont share an order with the trait genera and are not marine
    df_ref = df_ref[
        (~df_ref.worms_genus.isna())
        & (df_ref.worms_ismarine)
        & (df_ref.worms_order.isin(df_trait_genera.worms_order))
    ]

    genus_cov_index = pd.MultiIndex.from_frame(df_trait_genera).union(
        pd.MultiIndex.from_frame(df_ref[taxonomy_cols].drop_duplicates())
    )
    genus_cov = pd.DataFrame(index=genus_cov_index)
    genus_cov["traits_avail"] = genus_cov.index.get_level_values("worms_genus").isin(
        df_trait_genera.worms_genus
    )

    genus_cov.sort_values(
        by=["traits_avail"] + taxonomy_cols,
        ascending=[False] + [True for t in taxonomy_cols],
        inplace=True,
    )

    genus_cov["NCBI + SILVA"] = df_ref.groupby(taxonomy_cols).size().astype("Int64")

    # 3. Filter out short sequences
    df_ref = df_ref[df_ref.length >= cfg.min_length]

    genus_cov[f"Length >= {cfg.min_length}"] = df_ref.groupby(taxonomy_cols).size().astype("Int64")

    # 4. Filter out low quality sequences
    df_ref = df_ref[df_ref.ambiguity_frac <= cfg.max_ambiguity]

    genus_cov[f"Ambiguity <= {cfg.max_ambiguity}"] = (
        df_ref.groupby(taxonomy_cols).size().astype("Int64")
    )

    # 5. Filter out sequences that dont span the OTU frame
    df_ref = df_ref[df_ref.otu_coverage >= cfg.otu_coverage]

    genus_cov[f"OTU coverage >= {cfg.otu_coverage}"] = (
        df_ref.groupby(taxonomy_cols).size().astype("Int64")
    )

    # 6. Group by (species, cluster) and keep only the longest sequence
    df_ref = df_ref.sort_values(
        ["worms_name", "clust_id", "length", "otu_coverage", "ambiguity_frac"],
        ascending=[True, True, False, False, True],
    )
    df_ref = df_ref.groupby(["worms_name", "clust_id"], sort=False, group_keys=False).head(1)

    genus_cov["Species x global cluster"] = df_ref.groupby(taxonomy_cols).size().astype("Int64")

    # 7. Filter out sequences that are extermely divergent to the other sequences
    msa = skbio.TabularMSA(create_compressed_alignment(df_ref), index=df_ref.index)
    dist = align_dists(msa, metric=cfg.dist_model, gamma=cfg.model_gamma, shared_by_all=False)
    dist.data[dist.data < 0.0] = 0.0

    dist_data = dist.data.copy()
    np.fill_diagonal(dist_data, np.nan)

    median_dist = np.nanmedian(dist_data, axis=1)
    z_scores = (median_dist - median_dist.mean()) / median_dist.std()

    keep_indices = np.flatnonzero(z_scores <= cfg.max_z_score)
    df_ref = df_ref.iloc[keep_indices]
    dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index)))

    genus_cov[f"TN93 Z-score <= {cfg.max_z_score}"] = (
        df_ref.groupby(taxonomy_cols).size().astype("Int64")
    )

    # 8. Perform hierarchical clustering on every species/genus
    worms_names = df_ref.worms_name.to_numpy()
    worms_name_clusters = np.arange(len(worms_names), dtype=np.int16)
    worms_name_silhouette_coefs = np.full(len(worms_names), np.nan)
    for name in np.unique(worms_names):
        name_idx = np.flatnonzero(worms_names == name)

        if len(name_idx) < 2:
            continue

        name_dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index[name_idx])))
        name_linkage = linkage(name_dist.condensed_form(), method="average")

        labels = fcluster(name_linkage, cfg.worms_name_cluster_dist, criterion="distance")
        if 1 < len(np.unique(labels)) < len(name_idx):
            coefs = silhouette_samples(name_dist.data, labels, metric="precomputed")
        else:
            coefs = np.full(len(name_idx), np.nan)

        worms_name_clusters[name_idx] = labels
        worms_name_silhouette_coefs[name_idx] = coefs

        plot_name_tree(name_linkage, name, list(df_ref.index[name_idx]), 0.1, f"plots/{name}.png")

    df_ref["worms_name_cluster"] = worms_name_clusters
    df_ref["worms_name_silhouette_coefs"] = worms_name_silhouette_coefs

    df_ref["worms_name_long_enough"] = df_ref.length >= (
        df_ref.groupby(["worms_name", "worms_name_cluster"]).length.transform("max")
        * cfg.worms_name_length_frac
    )

    df_ref = df_ref.sort_values(
        [
            "worms_name",
            "worms_name_cluster",
            "worms_name_long_enough",
            "worms_name_silhouette_coefs",
            "length",
            "otu_coverage",
            "ambiguity_frac",
        ],
        ascending=[True, True, False, False, False, False, True],
    )

    df_ref = df_ref.groupby(
        ["worms_name", "worms_name_cluster"], sort=False, group_keys=False
    ).head(1)
    dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index)))

    genus_cov["worms name cluster reps"] = df_ref.groupby(taxonomy_cols).size().astype("Int64")

    # 9. Remove genus only sequences that are too similar to a species
    worms_genera = df_ref.worms_genus.to_numpy()
    worms_species_level = (df_ref.worms_rank == "Species").to_numpy()

    keep = np.ones(len(df_ref), dtype=bool)

    for genus in worms_genera:
        genus_idx = np.flatnonzero(worms_genera == genus)

        named_idx = genus_idx[worms_species_level[genus_idx]]
        if len(named_idx) == 0:
            continue

        unnamed_idx = genus_idx[~worms_species_level[genus_idx]]
        for i in unnamed_idx:
            nearest_named_dist = np.min(dist.data[i, named_idx])

            if nearest_named_dist <= cfg.worms_name_cluster_dist:
                keep[i] = False

    df_ref = df_ref.iloc[np.flatnonzero(keep)]

    genus_cov["filter genus redundancy"] = df_ref.groupby(taxonomy_cols).size().astype("Int64")

    genus_cov.to_csv(cfg.output_genus_cov)

    newick_compat = str.maketrans({" ": "_", "(": "", ")": ""})
    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{accession}_{row.worms_name.translate(newick_compat)}\n{seq}\n"
            for (accession, row), seq in zip(df_ref.iterrows(), create_compressed_alignment(df_ref))
        )


if __name__ == "__main__":
    main()
