import concurrent.futures
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from pastml.acr import acr
from pastml.tree import name_tree, read_tree
from src.common import parse_config


@dataclass
class Config:
    input_trait_data: Path
    input_ml_tree: Path
    input_bootstrap_trees: Path
    output_tree: Path
    output_tree_support: Path
    output_acr_dir: Path


def linear_shannon(row):
    k = len(row)
    e = sum([(1 / (1 - k) * p) - (1 / (1 - k)) if p > 1 / k else p for p in row])
    return e


def shuffle_leaves(tree):
    shuffled_tree = tree.copy()
    leaves = shuffled_tree.get_leaves()

    names = [leaf.name for leaf in leaves]
    random.shuffle(names)

    for leaf, new_name in zip(leaves, names):
        leaf.name = new_name

    return shuffled_tree


def process_single_trait(trait, cfg, df_leaves, ml_tree, bs_trees, leaf_names):
    print(f"Running ACR for {trait}")

    ml_acr_results = acr(ml_tree, df_leaves[[trait]])
    df_ml_probs: pd.DataFrame = ml_acr_results[0]["marginal_probabilities"].drop(leaf_names)
    df_ml_probs.reset_index(names="node", inplace=True)
    df_ml_probs["tree_type"] = "ml"
    df_ml_probs["tree_id"] = 0

    bs_acr_results = acr(bs_trees, df_leaves[[trait]])
    df_bs_probs: pd.DataFrame = bs_acr_results[0]["marginal_probabilities"].drop(leaf_names)
    df_bs_probs.reset_index(names="node", inplace=True)
    df_bs_probs["tree_type"] = "bs"
    nodes_per_bs = len(df_bs_probs) // len(bs_trees)
    df_bs_probs["tree_id"] = np.arange(len(df_bs_probs)) // nodes_per_bs

    null_bs_trees = [shuffle_leaves(tree) for tree in bs_trees]
    null_bs_acr_results = acr(null_bs_trees, df_leaves[[trait]])
    df_null_probs: pd.DataFrame = null_bs_acr_results[0]["marginal_probabilities"].drop(leaf_names)
    df_null_probs.reset_index(names="node", inplace=True)
    df_null_probs["tree_type"] = "null_bs"
    df_null_probs["tree_id"] = np.arange(len(df_null_probs)) // nodes_per_bs

    df_trait_probs = pd.concat([df_ml_probs, df_bs_probs, df_null_probs])
    df_trait_probs.set_index(["tree_type", "tree_id", "node"], inplace=True)
    df_trait_probs.sort_index(inplace=True)
    df_trait_probs["entropy"] = df_trait_probs.apply(linear_shannon, axis=1)
    df_trait_probs.to_csv(cfg.output_acr_dir / f"acr_{trait}.csv")


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df_trait_data = pd.read_csv(cfg.input_trait_data, index_col="worms_genus")
    df_trait_data = df_trait_data.iloc[:, -16:]

    ml_tree = read_tree(cfg.input_ml_tree)
    name_tree(ml_tree)

    with open(cfg.input_bootstrap_trees, "r") as f:
        nwks = f.readlines()
    bs_trees = [read_tree(nwk) for nwk in nwks]
    for tree in bs_trees:
        name_tree(tree)

    ml_tree.write(format=1, outfile=cfg.output_tree)
    df_support = pd.DataFrame(
        [
            {"node": node.name, "support": node.support}
            for node in ml_tree.traverse()
            if not (node.is_root() or node.is_leaf())
        ]
    )
    df_support.to_csv(cfg.output_tree_support, index=False)

    leaf_names = ml_tree.get_leaf_names()
    df_leaves = pd.DataFrame(leaf_names, columns=["leaf_name"])
    df_leaves["worms_genus"] = df_leaves.leaf_name.str.split("_").str[0]

    df_leaves = pd.merge(df_leaves, df_trait_data, on="worms_genus")
    df_leaves = df_leaves.drop(columns="worms_genus").set_index("leaf_name")

    worker_func = partial(
        process_single_trait,
        cfg=cfg,
        df_leaves=df_leaves,
        ml_tree=ml_tree,
        bs_trees=bs_trees,
        leaf_names=leaf_names,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker_func, df_leaves.columns))


if __name__ == "__main__":
    main()
