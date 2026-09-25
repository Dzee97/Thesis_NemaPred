from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import skbio
from src.common import parse_config


@dataclass
class Config:
    input_tsv: Path
    output_fasta: Path


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df = pd.read_csv(
        cfg.input_tsv,
        delimiter="\t",
        usecols=["sampleid", "genus", "species", "nuc", "marker_code"],
    )
    df = df[df.marker_code.str.startswith("18S")]
    df = df[~df.genus.isna()]

    print(f"Writing {len(df)} BOLD Nematode records to fasta file")

    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">BOLD|{row.sampleid} {row.genus};{row.species}\n{skbio.DNA(row.nuc).degap().transcribe()}\n"
            for _, row in df.iterrows()
        )


if __name__ == "__main__":
    main()
