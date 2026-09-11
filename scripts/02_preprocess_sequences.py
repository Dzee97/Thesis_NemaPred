import gzip
import hashlib
import heapq
import io
import tarfile
import zlib
import argparse
from array import array
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from sqlmodel import Field, Session, SQLModel, create_engine, select


class SequenceInfo(SQLModel, table=True):
    sequence_id: str = Field(primary_key=True)
    description: str
    aligned_zlib: bytes
    positions_bytes: bytes
    bases_bytes: bytes
    length: int
    ambiguity_fraction: float
    first_column: int
    last_column: int

    @property
    def positions(self) -> np.ndarray:
        return np.frombuffer(self.positions_bytes, dtype=np.int32)

    @property
    def bases(self) -> np.ndarray:
        return np.frombuffer(self.bases_bytes, dtype=np.uint8)

    @property
    def aligned_sequence(self) -> str:
        return zlib.decompress(self.aligned_zlib).decode("ascii")

    @property
    def ungapped_sequence(self) -> str:
        return self.bases.tobytes().decode("ascii")


def sequence_normalization_mapping() -> dict[int, str]:
    IUPAC_CHARS = "ACGTRYSWKMBDHVN"
    GAP_CHARS = "-.~_"

    trans_dict = {}
    for c in IUPAC_CHARS:
        trans_dict[ord(c)] = c
        trans_dict[ord(c.lower())] = c
    trans_dict[ord("U")] = "T"
    trans_dict[ord("u")] = "T"
    for c in GAP_CHARS:
        trans_dict[ord(c)] = "-"

    trans_dict.update({i: ord("N") for i in range(256) if i not in trans_dict})

    return trans_dict


def ambiguity_fraction(bases: np.ndarray) -> float:
    if len(bases) == 0:
        return 1.0

    is_unambiguous = (
        (bases == ord("A")) | (bases == ord("C")) | (bases == ord("G")) | (bases == ord("T"))
    )
    return 1.0 - float(is_unambiguous.mean())


def sparse_sequence(aligned_sequence: str) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.frombuffer(aligned_sequence.encode("ascii"), dtype=np.uint8)
    positions = np.flatnonzero(encoded != ord("-")).astype(np.int32)
    bases = encoded[positions].copy()
    return positions, bases


@dataclass
class Config:
    min_ungapped_length: int
    max_ambiguity_fraction: float
    input_fasta: Path
    output_aligned_fasta: Path
    output_ungapped_fasta: Path


def main() -> None:
    snakemake = globals().get("snakemake", None)

    if snakemake is not None:
        cfg = Config(**(dict(snakemake.input) | dict(snakemake.output) | dict(snakemake.params)))
    else:
        parser = argparse.ArgumentParser()
        for field in fields(Config):
            flag = f"--{field.name}"
            kwargs = {"type": field.type, "required": True}
            if field.type is bool:
                kwargs = {"action": "store_true", "required": False}

            parser.add_argument(flag, **kwargs)

        cfg = Config(**vars(parser.parse_args()))

    num_sequences = 0
    alignment_length = None
    normalize_mapping = sequence_normalization_mapping()

    sqlite_url = "sqlite:///sequences.db"
    batch_size = 1000
    engine = create_engine(sqlite_url)
    SQLModel.metadata.create_all(engine)

    with tarfile.open(cfg.input_fasta, "r:gz") as tar:
        member = tar.getmembers()[0]
        tar_file = tar.extractfile(member)
        if tar_file is None:
            raise TypeError(f"Could not extract data from: {member.name}")

        with (
            io.TextIOWrapper(tar_file, encoding="utf-8", errors="replace") as text_in,
            Session(engine) as session,
        ):
            for record in tqdm(SeqIO.parse(text_in, "fasta"), desc="Loading sequences"):
                num_sequences += 1

                if alignment_length is None:
                    alignment_length = len(record.seq)
                elif len(record.seq) != alignment_length:
                    raise ValueError(
                        f"Inconsistent alignment length for {record.id}: "
                        f"{len(record.seq)} != {alignment_length}"
                    )

                normalized_seq = str(record.seq).translate(normalize_mapping)

                positions, bases = sparse_sequence(normalized_seq)

                session.add(
                    SequenceInfo(
                        sequence_id=record.id,
                        description=record.description,
                        aligned_zlib=zlib.compress(normalized_seq.encode("ascii"), level=1),
                        positions_bytes=positions.tobytes(),
                        bases_bytes=bases.tobytes(),
                        length=len(positions),
                        ambiguity_fraction=ambiguity_fraction(bases),
                        first_column=int(positions[0]),
                        last_column=int(positions[-1]),
                    )
                )

                if num_sequences % batch_size == 0:
                    session.commit()

            session.commit()


if __name__ == "__main__":
    main()
