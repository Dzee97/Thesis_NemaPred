from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skbio
from matplotlib import colormaps
from matplotlib.colors import to_hex
from sklearn.metrics import silhouette_samples
from src.common import parse_config


@dataclass
class Config:
    input_parquet: Path
    input_trait_data: Path
    input_tree: Path
    output_fasta: Path
    output_selection: Path
    output_outgroup: Path
    num_genus_neighbors: int


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


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    pa_table = pq.read_table(cfg.input_parquet)
    df: pd.DataFrame = pa_table.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
    df.set_index("accession", inplace=True)

    df_ref = df[df.seq_type == "ref"].copy()
    df_out = df[df.seq_type == "outgroup"].copy()
    df_out.rename(columns={"worms_name": "tip_name"}, inplace=True)

    tree: skbio.TreeNode | Any
    tree = skbio.read(cfg.input_tree, format="newick", into=skbio.TreeNode)
    tree_dist = tree.cophenet()
    tree_dist.ids = [id.split()[0] for id in tree_dist.ids]
    tree_dist_data = tree_dist.data.copy()
    np.fill_diagonal(tree_dist_data, np.nan)

    df_ref = df_ref.loc[list(tree_dist.ids)]

    worms_genera = df_ref.worms_genus.to_numpy()
    unique_genera = np.unique(worms_genera)

    family_fraction = np.full(len(df_ref), np.nan)
    order_fraction = np.full(len(df_ref), np.nan)
    class_fraction = np.full(len(df_ref), np.nan)

    taxonomy_lookup = df_ref.drop_duplicates("worms_genus").set_index("worms_genus")[
        ["worms_family", "worms_order", "worms_class"]
    ]
    family_count = taxonomy_lookup.groupby("worms_family").size()
    order_count = taxonomy_lookup.groupby("worms_order").size()
    class_count = taxonomy_lookup.groupby("worms_class").size()

    for i in range(len(df_ref)):
        other_genus_info = []

        for other_genus in unique_genera:
            if other_genus == worms_genera[i]:
                continue

            other_genus_idx = np.flatnonzero(worms_genera == other_genus)
            other_genus_dist = np.nanmedian(tree_dist_data[i, other_genus_idx])
            other_genus_info.append((other_genus_dist, other_genus))

        other_genus_info.sort(key=lambda x: x[0])
        neighbors = other_genus_info[: cfg.num_genus_neighbors]

        neighbor_genera = [x[1] for x in neighbors]
        neighbor_taxonomy = taxonomy_lookup.loc[neighbor_genera]

        genus_taxonomy = taxonomy_lookup.loc[worms_genera[i]]

        genus_family = genus_taxonomy.worms_family
        if family_count[genus_family] > 1:
            family_fraction[i] = (neighbor_taxonomy.worms_family == genus_family).mean()

        genus_order = genus_taxonomy.worms_order
        if order_count[genus_order] > 1:
            order_fraction[i] = (neighbor_taxonomy.worms_order == genus_order).mean()

        genus_class = genus_taxonomy.worms_class
        if class_count[genus_class] > 1:
            class_fraction[i] = (neighbor_taxonomy.worms_class == genus_class).mean()

    df_ref["nearest_k_family_fraction"] = family_fraction
    df_ref["nearest_k_order_fraction"] = order_fraction
    df_ref["nearest_k_class_fraction"] = class_fraction

    df_trait_data = pd.read_csv(cfg.input_trait_data, usecols=["worms_genus"])

    df_ref = df_ref[df_ref.worms_genus.isin(df_trait_data.worms_genus)]

    # Remove sequences that dont share a class with any of the nearest other genera.
    # df_ref = df_ref[df_ref.nearest_k_class_fraction > 0]
    tree_dist = cast(skbio.DistanceMatrix, tree_dist.filter(list(df_ref.index)))
    df_ref["worms_family_silhoette_coef"] = np.asarray(
        silhouette_samples(tree_dist.data, df_ref.worms_family.to_numpy(), metric="precomputed")
    )
    df_ref["worms_family_silhoette_pos"] = df_ref.worms_family_silhoette_coef >= 0
    df_ref["worms_genus_silhoette_median"] = df_ref.groupby(
        "worms_genus"
    ).worms_family_silhoette_coef.transform("median")

    df_ref["keep"] = df_ref.worms_family_silhoette_coef >= (
        df_ref.worms_genus_silhoette_median - 0.5 * df_ref.worms_genus_silhoette_median.abs()
    )

    is_genus = df_ref.worms_rank == "Genus"
    multiple_seqs = df_ref.groupby("worms_name").worms_name.transform("count") > 1
    unassigned_nums = df_ref[(is_genus & multiple_seqs)].groupby("worms_name").cumcount() + 1
    df_ref["tip_name"] = df_ref["worms_name"].copy()
    df_ref.loc[is_genus, "tip_name"] += " sp."
    df_ref.loc[(is_genus & multiple_seqs), "tip_name"] += " " + unassigned_nums.astype(str)

    selection_cols = [
        "worms_class",
        "worms_order",
        "worms_family",
        "worms_genus",
        "tip_name",
        "nearest_k_family_fraction",
        "nearest_k_order_fraction",
        "nearest_k_class_fraction",
        "worms_family_silhoette_coef",
        "worms_genus_silhoette_median",
        "keep",
        "length",
        "otu_coverage",
        "ambiguity_frac",
    ]

    df_ref = df_ref.sort_values(
        selection_cols,
        ascending=[
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
    )

    df_ref = df_ref.groupby("tip_name", sort=False, group_keys=False).head(1)

    df_ref[selection_cols].to_csv(cfg.output_selection)

    df_alignment = df.loc[df_ref.index.union(df_out.index)][["rna_seq", "msa_pos"]]
    df_alignment["tip_name"] = pd.concat([df_ref.tip_name, df_out.tip_name])

    newick_compat = str.maketrans({" ": "_", "(": "", ")": ""})
    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{row.tip_name.translate(newick_compat)}\n{seq}\n"
            for (_, row), seq in zip(
                df_alignment.iterrows(), create_compressed_alignment(df_alignment)
            )
        )

    with open(cfg.output_outgroup, "w") as f:
        f.write(
            ",".join(f"{row.tip_name.translate(newick_compat)}" for _, row in df_out.iterrows())
        )


if __name__ == "__main__":
    main()
