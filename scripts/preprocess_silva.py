from dataclasses import dataclass
from pathlib import Path

import skbio
from common import parse_config


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
            fields = seq.metadata["description"].split(";")[-1].split()
            hyp_genus = fields[0]

            dna_seq = seq.reverse_transcribe()
            dna_seq = dna_seq.replace(dna_seq.gaps(), "-")

            f.write(f">{hyp_genus}\n{dna_seq}\n")
            count += 1

    print(f"Writing {count} SILVA Nematode records to fasta file")


if __name__ == "__main__":
    main()
