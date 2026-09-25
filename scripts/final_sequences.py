from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import skbio
from src.common import parse_config

from scripts.filter_sequences import create_compressed_alignment


@dataclass
class Config:
    input_parquet: Path
    input_tree: Path
    output_fasta: Path


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    pa_table = pq.read_table(cfg.input_parquet)
    df: pd.DataFrame = pa_table.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
    df.set_index("accession", inplace=True)

    tree: skbio.TreeNode | Any
    tree = skbio.read(cfg.input_tree, format="newick", into=skbio.TreeNode)
    tree_dist = tree.cophenet()
    tree_ids = [id.split()[0] for id in tree_dist.ids]
    tree_idx = {id: i for i, id in enumerate(tree_ids)}

    df = df.loc[tree_ids]
    df_ref = df[df.seq_type == "ref"].copy()
    df_out = df[df.seq_type == "outgroup"]

    idx_map = np.array([tree_idx[accession] for accession in df_ref.index])
    dist_data = tree_dist.data[np.ix_(idx_map, idx_map)]
    np.fill_diagonal(dist_data, np.nan)

    genera = df_ref.worms_genus.to_numpy()
    median_genus_dist = np.full(len(genera), np.nan)

    for genus in np.unique(genera):
        genus_idx = np.flatnonzero(genera == genus)

        if len(genus_idx) < 2:
            continue

        genus_dist = dist_data[np.ix_(genus_idx, genus_idx)]

        median_genus_dist[genus_idx] = np.nanmedian(genus_dist, axis=1)

    df_ref["median_genus_dist"] = median_genus_dist

    other_genus_dist = dist_data.copy()
    for i, genus in enumerate(genera):
        same_genus = genera == genus
        other_genus_dist[i, same_genus] = np.inf

    nearest_other_idx = np.argmin(other_genus_dist, axis=1)

    df_ref["nearest_other"] = df_ref.index[nearest_other_idx]
    df_ref["nearest_other_family"] = df_ref.nearest_other.map(df_ref.worms_family)
    df_ref["nearest_other_order"] = df_ref.nearest_other.map(df_ref.worms_order)

    df_ref["nearest_other_family_match"] = df_ref.worms_family == df_ref.nearest_other_family
    df_ref["nearest_other_order_match"] = df_ref.worms_order == df_ref.nearest_other_order

    df_ref = df_ref.sort_values(
        [
            "worms_genus",
            "nearest_other_order_match",
            "nearest_other_family_match",
            "median_genus_dist",
            "length",
            "ambiguity_frac",
        ],
        ascending=[True, False, False, True, False, True],
    )

    df_ref = df_ref.groupby("worms_genus", sort=False, group_keys=False).head(1)

    df_final = df.loc[df_ref.index.union(df_out.index)]

    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{accession}_{row.worms_genus}\n{seq}\n"
            for (accession, row), seq in zip(
                df_final.iterrows(), create_compressed_alignment(df_final)
            )
        )


if __name__ == "__main__":
    main()
