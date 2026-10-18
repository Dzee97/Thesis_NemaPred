import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skbio
from skbio.alignment import align_dists
from skbio.tree import nj, nni
from sklearn.metrics import silhouette_samples
from src.common import parse_config


@dataclass
class Config:
    input_parquet: Path
    input_trait_data: Path
    output_raxml_dir: Path
    output_fasta: Path
    output_genus_cov: Path
    min_length: int
    max_ambiguity: float
    otu_coverage: float
    dist_model: str
    model_gamma: float
    max_z_score: float
    raxml_model: str


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


def mark_redundant(df_ref: pd.DataFrame, dist_data: np.ndarray) -> np.ndarray:
    worms_genus = df_ref.worms_genus.to_numpy()
    worms_name = df_ref.worms_name.to_numpy()
    worms_rank = df_ref.worms_rank.to_numpy()

    sil_coefs = np.asarray(silhouette_samples(dist_data, worms_genus, metric="precomputed"))

    dist_data = dist_data.copy()
    np.fill_diagonal(dist_data, np.nan)

    keep = np.ones(len(df_ref), dtype=bool)
    iter_num = 0

    while True:
        keep_idx = np.flatnonzero(keep)
        to_remove = set()

        for idx in keep_idx:
            if idx in to_remove:
                continue

            local_closest = np.nanargmin(dist_data[idx, keep_idx])
            closest_idx = keep_idx[local_closest]

            if worms_genus[idx] == worms_genus[closest_idx]:
                if worms_name[idx] == worms_name[closest_idx]:
                    worst_idx = min(closest_idx, idx, key=lambda x: sil_coefs[x])
                elif worms_rank[idx] != worms_rank[closest_idx]:
                    worst_idx = idx if worms_rank[idx] == "Genus" else closest_idx
                else:
                    continue

                to_remove.add(worst_idx)

        if to_remove:
            keep[list(to_remove)] = False
            print(f"Mark {len(to_remove)}, keep {sum(keep)}")
        else:
            break

        iter_num += 1

    return keep


def remove_redundant(
    df_ref: pd.DataFrame, dist: skbio.DistanceMatrix, on_tree=False
) -> tuple[pd.DataFrame, skbio.DistanceMatrix]:
    iter_count = 0

    while True:
        iter_count += 1

        if on_tree:
            print(f"Remove redundant on NJ cophenet - iter {iter_count}")

            tree = nj(dist)
            tree_dist = tree.cophenet()
            tree_dist = tree_dist.filter(list(df_ref.index))
            dist_data = tree_dist.data
        else:
            print(f"Remove redundant on TN93 dist - iter {iter_count}")

            dist_data = dist.data

        keep = mark_redundant(df_ref, dist_data)

        if sum(keep) == len(df_ref):
            return df_ref, dist

        df_ref = df_ref.loc[keep]
        dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index)))


def remove_family_outliers(
    df_ref: pd.DataFrame, dist: skbio.DistanceMatrix, silhoette_margin=0.3, min_genera_per_family=3
) -> tuple[pd.DataFrame, skbio.DistanceMatrix]:

    worms_family = df_ref.worms_family.to_numpy()
    worms_genus = df_ref.worms_genus.to_numpy()

    family_silhoette_coef = np.asarray(
        silhouette_samples(dist.data, worms_family, metric="precomputed")
    )
    genus_silhoette_coef = np.asarray(
        silhouette_samples(dist.data, worms_genus, metric="precomputed")
    )

    family_silhoette_median = np.full(len(worms_family), np.nan)
    family_testable = np.zeros(len(df_ref), dtype=bool)

    for family in np.unique(worms_family):
        family_idx = np.flatnonzero(worms_family == family)

        family_genera = np.unique(worms_genus[family_idx])
        if len(family_genera) < min_genera_per_family:
            continue

        genus_medians = []
        for genus in family_genera:
            genus_idx = np.flatnonzero((worms_family == family) & (worms_genus == genus))
            genus_medians.append(np.median(family_silhoette_coef[genus_idx]))

        family_silhoette_median[family_idx] = np.median(genus_medians)
        family_testable[family_idx] = True

    family_outlier = (
        family_testable
        & (family_silhoette_coef < 0)
        & (family_silhoette_coef < family_silhoette_median - silhoette_margin)
    )

    genus_size = df_ref.groupby("worms_genus").worms_genus.transform("size").to_numpy()
    genus_outlier = (genus_size > 1) & (genus_silhoette_coef < 0)

    remove = family_outlier & genus_outlier

    print(f"Removed {sum(remove)} family outliers, keep {sum(~remove)}")

    df_ref = df_ref[~remove]
    dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index)))

    return df_ref, dist


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    pa_table = pq.read_table(cfg.input_parquet)
    df: pd.DataFrame = pa_table.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
    df.set_index("accession", inplace=True)

    df_ref = df[df.seq_type == "ref"]

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

    current_len = len(df_ref)
    iter_count = 0

    while True:
        iter_count += 1

        # 8. Iterively remove sequences based on close neighbor same genus agreement
        df_ref, dist = remove_redundant(df_ref, dist, on_tree=False)

        genus_cov[f"I{iter_count} - TN93 genus redundancy"] = (
            df_ref.groupby(taxonomy_cols).size().astype("Int64")
        )

        # 9. Perform NJ and again remove redundant sequences
        df_ref, dist = remove_redundant(df_ref, dist, on_tree=True)

        genus_cov[f"I{iter_count} - NJ genus redundancy"] = (
            df_ref.groupby(taxonomy_cols).size().astype("Int64")
        )

        # 10. Filter outliers by family silhoutte scores
        df_ref, dist = remove_family_outliers(df_ref, dist)

        genus_cov[f"I{iter_count} - Family outliers"] = (
            df_ref.groupby(taxonomy_cols).size().astype("Int64")
        )

        if len(df_ref) == current_len:
            break

        current_len = len(df_ref)

    # 11. Limit to target genera and again remove outliers
    df_ref = df_ref[df_ref.worms_genus.isin(df_trait_genera.worms_genus)]
    dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index)))

    df_ref, dist = remove_redundant(df_ref, dist, on_tree=False)

    genus_cov["Trait only - TN93 genus redundancy"] = (
        df_ref.groupby(taxonomy_cols).size().astype("Int64")
    )

    df_ref, dist = remove_redundant(df_ref, dist, on_tree=True)

    genus_cov["Trait only - NJ genus redundancy"] = (
        df_ref.groupby(taxonomy_cols).size().astype("Int64")
    )

    genus_cov.to_csv(cfg.output_genus_cov)

    newick_compat = str.maketrans({" ": "_", "(": "", ")": ""})
    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{accession}_{row.worms_name.translate(newick_compat)}\n{seq}\n"
            for (accession, row), seq in zip(df_ref.iterrows(), create_compressed_alignment(df_ref))
        )


if __name__ == "__main__":
    main()
