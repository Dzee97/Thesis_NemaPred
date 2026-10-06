import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import skbio
from matplotlib import colormaps
from matplotlib.colors import to_hex
from src.common import parse_config

from scripts.filter_sequences import create_compressed_alignment


@dataclass
class Config:
    input_parquet: Path
    input_trait_genera: Path
    input_tree: Path
    output_fasta: Path
    output_selection: Path
    output_outgroup: Path
    output_itol_class: Path
    output_itol_order: Path
    output_itol_family_dir: Path
    num_genus_neighbors: int


def make_tab10_colors(labels):
    labels = list(labels)
    cmap = colormaps["tab10"]

    return {label: to_hex(cmap(i % 10)) for i, label in enumerate(labels)}


def write_itol_colorstrip(
    df: pd.DataFrame,
    taxon_col: str,
    dataset_label: str,
    output_path: Path,
):
    values = sorted(df[taxon_col].dropna().unique())

    colors = make_tab10_colors(values)

    with open(output_path, "w") as f:
        f.write("DATASET_COLORSTRIP\n")
        f.write("SEPARATOR TAB\n")
        f.write(f"DATASET_LABEL\t{dataset_label}\n")
        f.write("COLOR\t#000000\n")

        if values:
            f.write(f"LEGEND_TITLE\t{dataset_label}\n")
            f.write("LEGEND_SHAPES\t" + "\t".join(["1"] * len(values)) + "\n")
            f.write("LEGEND_COLORS\t" + "\t".join(colors[v] for v in values) + "\n")
            f.write("LEGEND_LABELS\t" + "\t".join(str(v) for v in values) + "\n")

        f.write("DATA\n")

        for _, row in df.iterrows():
            value = row[taxon_col]

            if pd.isna(value):
                continue

            tip_name = f"{row.worms_genus}"

            f.write(f"{tip_name}\t{colors[value]}\t{value}\n")


def write_itol_family_strips_per_order(
    df: pd.DataFrame,
    output_dir: Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # only rows that have both order and family
    df_family = df[df.worms_order.notna() & df.worms_family.notna()].copy()

    for order, df_order in df_family.groupby("worms_order", sort=True):
        families = sorted(df_order.worms_family.unique())

        if len(families) == 0:
            continue

        filename = output_dir / f"itol_family_{order}.txt"

        write_itol_colorstrip(
            df=df_order,
            taxon_col="worms_family",
            dataset_label=f"Family ({order})",
            output_path=filename,
        )


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

    df_out = df[df.seq_type == "outgroup"]
    df_ref = df[df.seq_type == "ref"].copy()

    df_ref = df_ref.loc[tree_ids]

    idx_map = np.array([tree_idx[accession] for accession in df_ref.index])
    dist_data = tree_dist.data[np.ix_(idx_map, idx_map)]
    np.fill_diagonal(dist_data, np.nan)

    genera = df_ref.worms_genus.to_numpy()
    unique_genera = np.unique(genera)

    family_fraction = np.full(len(df_ref), np.nan)
    order_fraction = np.full(len(df_ref), np.nan)

    taxonomy_lookup = df_ref.drop_duplicates("worms_genus").set_index("worms_genus")[
        ["worms_family", "worms_order"]
    ]
    family_count = taxonomy_lookup.groupby("worms_family").size()
    order_count = taxonomy_lookup.groupby("worms_order").size()

    for i, genus in enumerate(genera):
        other_genus_info = []

        for other_genus in unique_genera:
            if other_genus == genus:
                continue

            other_genus_idx = np.flatnonzero(genera == other_genus)
            other_genus_dist = np.nanmedian(dist_data[i, other_genus_idx])
            other_genus_info.append((other_genus_dist, other_genus))

        other_genus_info.sort(key=lambda x: x[0])
        neighbors = other_genus_info[: cfg.num_genus_neighbors]

        neighbor_genera = [x[1] for x in neighbors]
        neighbor_taxonomy = taxonomy_lookup.loc[neighbor_genera]

        genus_taxonomy = taxonomy_lookup.loc[genus]

        genus_family = genus_taxonomy.worms_family
        if family_count[genus_family] > 1:
            family_fraction[i] = (neighbor_taxonomy.worms_family == genus_family).mean()

        genus_order = genus_taxonomy.worms_order
        if order_count[genus_order] > 1:
            order_fraction[i] = (neighbor_taxonomy.worms_order == genus_order).mean()

    df_ref["nearest_k_family_fraction"] = family_fraction
    df_ref["nearest_k_order_fraction"] = order_fraction

    df_genera = pd.read_excel(cfg.input_trait_genera)
    df_genera.set_index("Genus", inplace=True)

    df_ref = df_ref[df_ref.worms_genus.isin(df_genera.index)]

    selection_cols = [
        "worms_genus",
        "nearest_k_order_fraction",
        "nearest_k_family_fraction",
        "length",
        "ambiguity_frac",
    ]

    df_ref = df_ref.sort_values(
        selection_cols,
        ascending=[True, False, False, False, True],
    )

    df_ref = df_ref.groupby("worms_genus", sort=False, group_keys=False).head(1)

    df_ref[selection_cols].to_csv(cfg.output_selection)

    df_final = df.loc[df_ref.index.union(df_out.index)]

    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">{row.worms_genus}\n{seq}\n"
            for (_, row), seq in zip(df_final.iterrows(), create_compressed_alignment(df_final))
        )

    with open(cfg.output_outgroup, "w") as f:
        f.write(",".join(f"{row.worms_genus}" for _, row in df_out.iterrows()))

    write_itol_colorstrip(
        df_ref,
        "worms_class",
        "Class",
        cfg.output_itol_class,
    )

    write_itol_colorstrip(
        df_ref,
        "worms_order",
        "Order",
        cfg.output_itol_order,
    )

    write_itol_family_strips_per_order(
        df_ref,
        cfg.output_itol_family_dir,
    )


if __name__ == "__main__":
    main()
