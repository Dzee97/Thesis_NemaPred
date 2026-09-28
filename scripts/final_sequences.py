import colorsys
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
    input_trait_genera: Path
    input_tree: Path
    output_fasta: Path
    output_selection: Path
    output_itol_class: Path
    output_itol_order: Path
    output_itol_family: Path


def write_itol_colorstrip(
    df: pd.DataFrame,
    taxon_col: str,
    dataset_label: str,
    output_path: Path,
):
    # Only taxa with an assigned value
    taxa = sorted(df[taxon_col].dropna().unique())

    # Generate one deterministic color per taxon
    colors = {}

    for i, taxon in enumerate(taxa):
        hue = i / len(taxa)

        r, g, b = colorsys.hsv_to_rgb(
            hue,
            0.65,
            0.90,
        )

        colors[taxon] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    with open(output_path, "w") as f:
        f.write("DATASET_COLORSTRIP\n")
        f.write("SEPARATOR TAB\n")
        f.write(f"DATASET_LABEL\t{dataset_label}\n")
        f.write("COLOR\t#000000\n")

        # Legend
        if taxa:
            f.write(f"LEGEND_TITLE\t{dataset_label}\n")
            f.write("LEGEND_SHAPES\t" + "\t".join(["1"] * len(taxa)) + "\n")
            f.write("LEGEND_COLORS\t" + "\t".join(colors[taxon] for taxon in taxa) + "\n")
            f.write("LEGEND_LABELS\t" + "\t".join(str(taxon) for taxon in taxa) + "\n")

        f.write("DATA\n")

        for accession, row in df.iterrows():
            taxon = row[taxon_col]

            if pd.isna(taxon):
                continue

            tip_name = f"{accession}_{row.worms_genus}"

            f.write(f"{tip_name}\t{colors[taxon]}\t{taxon}\n")


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
    best_genus_dist = np.full(len(genera), np.nan)

    for genus in np.unique(genera):
        genus_idx = np.flatnonzero(genera == genus)

        if len(genus_idx) < 2:
            continue

        genus_dist = dist_data[np.ix_(genus_idx, genus_idx)]
        median_dist = np.nanmedian(genus_dist, axis=1)

        median_genus_dist[genus_idx] = median_dist
        best_genus_dist[genus_idx] = np.min(median_dist)

    _, genus_first_idx = np.unique(genera, return_index=True)
    eps_values = best_genus_dist[genus_first_idx]
    eps = np.nanquantile(eps_values, 0.2)

    df_ref["central"] = np.isnan(median_genus_dist) | (
        median_genus_dist <= best_genus_dist + 0.5 * np.maximum(best_genus_dist, eps)
    )

    keep_indices = np.flatnonzero(df_ref.central.to_numpy())
    dist_data = dist_data[np.ix_(keep_indices, keep_indices)]
    df_ref = df_ref[df_ref.central]

    genera = df_ref.worms_genus.to_numpy()
    families = df_ref.worms_family.to_numpy()
    orders = df_ref.worms_order.to_numpy()

    unique_genera = np.unique(genera)

    family_fraction = np.full(len(df_ref), np.nan)
    order_fraction = np.full(len(df_ref), np.nan)

    for i, genus in enumerate(genera):
        other_genus_info = []

        for other_genus in unique_genera:
            if other_genus == genus:
                continue

            genus_idx = np.flatnonzero(genera == other_genus)

            local_idx = np.nanargmin(dist_data[i, genus_idx])
            closest_tip_idx = genus_idx[local_idx]

            min_dist = dist_data[i, closest_tip_idx]

            other_genus_info.append((min_dist, other_genus, closest_tip_idx))

        other_genus_info.sort(key=lambda x: x[0])
        neighbors = other_genus_info[:5]

        neighbor_idx = np.array([x[2] for x in neighbors])

        other_family_exists = np.any((genera != genus) & (families == families[i]))
        if other_family_exists:
            family_fraction[i] = np.mean(families[neighbor_idx] == families[i])

        other_order_exists = np.any((genera != genus) & (orders == orders[i]))
        if other_order_exists:
            order_fraction[i] = np.mean(orders[neighbor_idx] == orders[i])

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
            f">{accession}_{row.worms_genus}\n{seq}\n"
            for (accession, row), seq in zip(
                df_final.iterrows(), create_compressed_alignment(df_final)
            )
        )

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

    write_itol_colorstrip(
        df_ref,
        "worms_family",
        "Family",
        cfg.output_itol_family,
    )


if __name__ == "__main__":
    main()
