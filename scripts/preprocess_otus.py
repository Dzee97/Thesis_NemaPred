from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import skbio
from src.common import parse_config


@dataclass
class Config:
    input_otus: Path
    output_fasta: Path


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df = pd.read_excel(cfg.input_otus, sheet_name="filtered_OTU_table", usecols=["sequence"])

    with open(cfg.output_fasta, "w") as f:
        f.writelines(
            f">OTU{idx}\n{skbio.DNA(seq).transcribe()}\n" for idx, seq in df.sequence.items()
        )


if __name__ == "__main__":
    main()
