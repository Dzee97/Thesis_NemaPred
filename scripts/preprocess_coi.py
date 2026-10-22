from dataclasses import dataclass
from pathlib import Path

import skbio
from src.common import parse_config


@dataclass
class Config:
    input_fasta: Path
    output_template_fasta: Path
    output_fragments_fasta: Path


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    count_temp = 0
    count_frag = 0
    with (
        open(cfg.output_template_fasta, "w") as f_temp,
        open(cfg.output_fragments_fasta, "w") as f_frag,
    ):
        seq: skbio.DNA
        for seq in skbio.read(cfg.input_fasta, format="fasta", constructor=skbio.DNA):
            id = seq.metadata["id"]
            fields = seq.metadata["description"].split()
            hyp_genus = fields[0]
            hyp_species = " ".join(fields[:2]) if len(fields) > 1 else None

            if len(seq) > 10_000:
                f = f_temp
                count_temp += 1
            else:
                f = f_frag
                count_frag += 1

            f.write(f">NCBI|{id} {hyp_genus};{hyp_species}\n{seq.degap().transcribe()}\n")

    print(f"Writing {count_temp} NCBI Nematode template records to fasta file")
    print(f"Writing {count_frag} NCBI Nematode fragment records to fasta file")


if __name__ == "__main__":
    main()
