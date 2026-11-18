from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skbio
from common import parse_config
from process_alignments2 import load_seq_dict
from sklearn.metrics import silhouette_samples


@dataclass
class Config:
    input_fasta: Path
    input_tree: Path
    input_metadata: Path
    output_fasta: Path


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    pivot_df = pd.read_csv(cfg.input_metadata, header=[0, 1])
    pivot_df = pivot_df.rename(columns=lambda x: "" if x.startswith("Unnamed") else x, level=1)

    tree: skbio.TreeNode = skbio.read(
        cfg.input_tree, format="newick", into=skbio.TreeNode, convert_underscores=False
    )  # type: ignore
    tree_dist = tree.cophenet().filter(pivot_df.name.to_list())

    family_col = pivot_df.worms_family.to_numpy()
    pivot_df["family_silhouette"] = np.asarray(
        silhouette_samples(tree_dist.data, family_col, metric="precomputed")
    )

    pivot_df = pivot_df[pivot_df.trait_genus]

    genus_max_idx = pivot_df.groupby("worms_genus", sort=False).family_silhouette.idxmax()
    pivot_df = pivot_df.loc[genus_max_idx]

    seq_dict = load_seq_dict(cfg.input_fasta, skbio.DNA)
    seq_records = [seq_dict[id] for id in pivot_df.name]

    with open(cfg.output_fasta, "w") as f:
        f.writelines(f">{name}\n{seq}\n" for name, seq in zip(pivot_df.name, seq_records))


if __name__ == "__main__":
    main()
