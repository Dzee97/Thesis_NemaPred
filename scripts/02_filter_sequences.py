from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from src.common import parse_config
from tqdm import tqdm


@dataclass
class Config:
    input_parquet: Path
    output_fasta: Path
    allow_duplicates: bool
    allow_contained: bool
    top_k_species_length: int
    top_k_genus_length: int


@dataclass
class SequenceInfo:
    accession: str
    description: str

    positions_bytes: bytes
    bases_bytes: bytes

    length: int
    ambiguity_fraction: float
    first_column: int
    last_column: int

    worms_aphiaid: str | None = None
    worms_rank: str | None = None
    worms_name: str | None = None
    worms_genus: str | None = None
    worms_ismarine: bool | None = None

    worms_valid_name: str | None = None
    worms_valid_genus: str | None = None

    duplicate_of: str | None = None
    contained_by: str | None = None

    @property
    def fasta_header(self) -> str:
        return f"{self.accession}_{self.worms_valid_genus} {self.description}"

    @property
    def positions(self) -> np.ndarray:
        return np.frombuffer(self.positions_bytes, dtype=np.uint16)

    @property
    def bases(self) -> np.ndarray:
        return np.frombuffer(self.bases_bytes, dtype=np.uint8)

    @property
    def aligned_sequence(self) -> str:
        seq = np.full(50000, ord("-"), dtype=np.uint8)
        seq[self.positions] = self.bases
        return seq.tobytes().decode("ascii")


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df = pd.read_parquet(cfg.input_parquet)

    if not cfg.allow_duplicates:
        df = df[df["duplicate_of"].isna()]
    if not cfg.allow_contained:
        df = df[df["contained_by"].isna()]

    species_mask = df["worms_rank"] == "Species"
    species_ranks = (
        df[species_mask].groupby("worms_valid_name")["length"].rank(ascending=False, method="first")
    )

    genus_mask = df["worms_rank"] == "Genus"
    genus_ranks = (
        df[genus_mask].groupby("worms_valid_genus")["length"].rank(ascending=False, method="first")
    )

    keep_species = species_ranks <= cfg.top_k_species_length
    keep_genus = genus_ranks <= cfg.top_k_genus_length

    df = df[
        keep_species.reindex(df.index, fill_value=False)
        | keep_genus.reindex(df.index, fill_value=False)
    ]

    with open(cfg.output_fasta, "w") as f:
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Writing filtered sequences to FASTA"
        ):
            seq = SequenceInfo(**row)
            f.write(f">{seq.fasta_header}\n{seq.aligned_sequence}\n")


if __name__ == "__main__":
    main()
