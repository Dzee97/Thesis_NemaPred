import zlib

import numpy as np
from sqlmodel import Field, SQLModel


class SequenceInfo(SQLModel, table=True):
    sequence_id: str = Field(primary_key=True)

    tax_phylum: str
    tax_class: str
    tax_order: str
    tax_species: str

    aligned_zlib: bytes
    positions_bytes: bytes
    bases_bytes: bytes

    length: int
    ambiguity_fraction: float
    first_column: int
    last_column: int

    duplicate_of: str | None = Field(
        default=None,
        foreign_key="sequenceinfo.sequence_id",
        index=True,
    )

    contained_by: str | None = Field(
        default=None,
        foreign_key="sequenceinfo.sequence_id",
        index=True,
    )

    @property
    def positions(self) -> np.ndarray:
        return np.frombuffer(self.positions_bytes, dtype=np.uint16)

    @property
    def bases(self) -> np.ndarray:
        return np.frombuffer(self.bases_bytes, dtype=np.uint8)

    @property
    def aligned_sequence(self) -> str:
        return zlib.decompress(self.aligned_zlib).decode("ascii")

    @property
    def ungapped_sequence(self) -> str:
        return self.bases_bytes.decode("ascii")
