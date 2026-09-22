from dataclasses import dataclass

import numpy as np


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

    cluster_centroid: bool
    cluster_id: int
    cluster_pct_identity: float
    cluster_target: str
    cluster_duplicate: bool
    cluster_contained: bool

    worms_aphiaid: str | None = None
    worms_rank: str | None = None
    worms_name: str | None = None
    worms_genus: str | None = None
    worms_ismarine: bool | None = None

    worms_valid_name: str | None = None
    worms_valid_genus: str | None = None

    avg_tip_length: float | None = None

    @property
    def fasta_header(self) -> str:
        tax = "" if self.worms_valid_genus is None else f"_{self.worms_valid_genus}"
        return f"{self.accession}{tax} {self.description}"

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
