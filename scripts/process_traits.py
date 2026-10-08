from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from process_sequences import get_worms_taxonomy, load_worms_cache
from src.common import parse_config


@dataclass
class Config:
    input_trait_genera: Path
    input_trait_data: Path
    output_trait_data: Path
    worms_cache: Path


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df_trait_data = pd.read_excel(cfg.input_trait_data)

    worms_cache = load_worms_cache(cfg.worms_cache)

    worms_genus = []
    worms_family = []
    worms_order = []
    worms_class = []

    for genus in df_trait_data.Genus:
        worms_record = get_worms_taxonomy([genus.strip()], worms_cache)
        if worms_record is None:
            raise RuntimeError(f"No WoRMS record found for {genus}")
        if (
            worms_record["valid_name"] is not None
            and worms_record["scientificname"] != worms_record["valid_name"]
        ):
            new_worms_record = get_worms_taxonomy([worms_record["valid_name"]], worms_cache)
            worms_record = new_worms_record if new_worms_record is not None else worms_record

        worms_genus.append(worms_record["genus"])
        worms_family.append(worms_record["family"])
        worms_order.append(worms_record["order"])
        worms_class.append(worms_record["class"])

    df_trait_data["worms_genus"] = worms_genus
    df_trait_data["worms_family"] = worms_family
    df_trait_data["worms_order"] = worms_order
    df_trait_data["worms_class"] = worms_class

    df_trait_data.set_index(
        ["worms_class", "worms_order", "worms_family", "worms_genus"], inplace=True
    )
    df_trait_data.sort_index(inplace=True)
    df_trait_data.drop(columns="Genus", inplace=True)
    df_trait_data.columns = (
        df_trait_data.columns.str.lower().str.replace(" ", "_").str.replace("/", "-")
    )

    df_trait_data.to_csv(cfg.output_trait_data)


if __name__ == "__main__":
    main()
