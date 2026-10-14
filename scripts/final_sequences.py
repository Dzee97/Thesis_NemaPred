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

    selection_cols = [
        "worms_rank",
        "worms_class",
        "worms_order",
        "worms_family",
        "worms_genus",
        "worms_name",
        "length",
        "otu_coverage",
        "ambiguity_frac",
    ]

    df_ref = df[df.seq_type == "ref"][selection_cols].copy()
    df_out = df[df.seq_type == "outgroup"][selection_cols].copy()

    tree: skbio.TreeNode | Any
    tree = skbio.read(cfg.input_tree, format="newick", into=skbio.TreeNode)
    tree_dist = tree.cophenet()
    tree_dist.ids = [id.split()[0] for id in tree_dist.ids]
    tree_dist_data = tree_dist.data.copy()
    np.fill_diagonal(tree_dist_data, np.nan)

    df_ref = df_ref.loc[list(tree_dist.ids)]

    df_ref["worms_family_silhoette_coef"] = np.asarray(
        silhouette_samples(tree_dist.data, df_ref.worms_family.to_numpy(), metric="precomputed")
    )
    df_ref["worms_family_silhoette_genus_median"] = df_ref.groupby(
        "worms_genus"
    ).worms_family_silhoette_coef.transform("median")

    df_ref["worms_family_silhoette_genus_central"] = df_ref.worms_family_silhoette_coef >= (
        df_ref.worms_family_silhoette_genus_median - 0.3
    )

    df_trait_data = pd.read_csv(cfg.input_trait_data, usecols=["worms_genus"])
    df_ref = df_ref[df_ref.worms_genus.isin(df_trait_data.worms_genus)]

    df_ref = df_ref.sort_values(
        [
            "worms_class",
            "worms_order",
            "worms_family",
            "worms_genus",
            "worms_name",
            "worms_family_silhoette_genus_central",
            "length",
            "otu_coverage",
            "ambiguity_frac",
        ],
        ascending=[True, True, True, True, True, False, False, False, True],
    )

    is_genus = df_ref.worms_rank == "Genus"
    group_rank = df_ref.groupby("worms_name", sort=False, group_keys=False).cumcount()

    df_ref["keep"] = df_ref.worms_family_silhoette_genus_central & (
        is_genus | (~is_genus & (group_rank == 0))
    )

    df_keep = df_ref[df_ref["keep"]]
    is_genus = df_keep.worms_rank == "Genus"
    multiple_seqs = df_keep.groupby("worms_name").worms_name.transform("count") > 1
    unassigned_nums = df_keep[(is_genus & multiple_seqs)].groupby("worms_name").cumcount() + 1

    tip_name = df_keep["worms_name"].copy()
    tip_name[is_genus] += " sp. "
    tip_name[(is_genus & multiple_seqs)] += " " + unassigned_nums.astype(str)

    df_ref.loc[df_ref["keep"], "tip_name"] = tip_name

    df_ref.to_csv(cfg.output_selection)

    df_ref = df_ref[df_ref["keep"]]

    df_alignment = df.loc[df_ref.index.union(df_out.index)][["rna_seq", "msa_pos"]]
    df_alignment["tip_name"] = pd.concat([df_ref.tip_name, df_out.worms_name])

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
            ",".join(f"{row.worms_name.translate(newick_compat)}" for _, row in df_out.iterrows())
        )


if __name__ == "__main__":
    main()
