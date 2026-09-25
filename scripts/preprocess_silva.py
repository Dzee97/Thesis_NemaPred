from dataclasses import dataclass
from pathlib import Path

import skbio
from src.common import parse_config


@dataclass
class Config:
    input_fasta: Path
    output_fasta: Path


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    count = 0
    with open(cfg.output_fasta, "w") as f:
        seq: skbio.RNA
        for seq in skbio.read(cfg.input_fasta, format="fasta", constructor=skbio.RNA):
            id = seq.metadata["id"]
            fields = seq.metadata["description"].split(";")[-1].split()
            hyp_genus = fields[0]
            hyp_species = " ".join(fields[:2]) if len(fields) > 1 else None

            f.write(f">SILVA|{id} {hyp_genus};{hyp_species}\n{seq}\n")
            count += 1

    print(f"Writing {count} SILVA Nematode records to fasta file")


if __name__ == "__main__":
    main()
