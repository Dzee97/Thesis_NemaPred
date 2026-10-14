from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pastml import col_name2cat
from pastml.acr import acr
from pastml.annotation import preannotate_forest
from pastml.tree import name_tree, read_tree
from src.common import parse_config


@dataclass
class Config:
    input_trait_data: Path
    input_tree: Path
    output_tree: Path
    output_tree_support: Path
    output_acr_dir: Path


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df_trait_data = pd.read_csv(cfg.input_trait_data, index_col="worms_genus")
    df_trait_data = df_trait_data.iloc[:, -16:]

    tree = read_tree(cfg.input_tree)
    name_tree(tree)

    tree.write(format=1, outfile=cfg.output_tree)
    df_support = pd.DataFrame(
        [
            {"node": node.name, "support": node.support}
            for node in tree.traverse()
            if not (node.is_root() or node.is_leaf())
        ]
    )
    df_support.to_csv(cfg.output_tree_support, index=False)

    leaf_names = tree.get_leaf_names()
    df_leaves = pd.DataFrame(leaf_names, columns=["leaf_name"])
    df_leaves["worms_genus"] = df_leaves.leaf_name.str.split("_").str[0]

    df_leaves = pd.merge(df_leaves, df_trait_data, on="worms_genus")
    df_leaves = df_leaves.drop(columns="worms_genus").set_index("leaf_name")

    for trait in df_leaves.columns:
        print(f"Running ACR for {trait}")
        acr_results = acr(tree, df_leaves[[trait]])
        df_probs: pd.DataFrame = acr_results[0]["marginal_probabilities"].drop(leaf_names)
        df_probs.index.name = "node"
        df_probs.to_csv(cfg.output_acr_dir / f"acr_{trait}.csv")


if __name__ == "__main__":
    main()
