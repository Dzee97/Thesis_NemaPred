from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skbio
from filter_sequences import mark_redundant
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

    tree: skbio.TreeNode | Any
    tree = skbio.read(cfg.input_tree, format="newick", into=skbio.TreeNode)
    tree_dist = tree.cophenet()
    tree_dist.ids = [id.split()[0] for id in tree_dist.ids]

    df_ref = df_ref.loc[list(tree_dist.ids)]

    keep = mark_redundant(df_ref, tree_dist.data)

    df_ref = df_ref.loc[keep]

    is_genus = df_ref.worms_rank == "Genus"
    multiple_seqs = df_ref.groupby("worms_name").worms_name.transform("count") > 1
    unassigned_nums = df_ref[multiple_seqs].groupby("worms_name").cumcount() + 1

    tip_name = df_ref["worms_name"].copy()
    tip_name[is_genus] += " sp."
    tip_name[multiple_seqs] += " " + unassigned_nums.astype(str)

    df_ref.to_csv(cfg.output_selection)

    df_alignment = df.loc[df_ref.index.union(df_out.index)]
    df_alignment["tip_name"] = pd.concat([tip_name, df_out.worms_name])

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
