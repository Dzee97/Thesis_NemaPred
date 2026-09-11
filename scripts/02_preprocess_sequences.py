import hashlib
import io
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import SeqIO
from sqlmodel import Session, SQLModel, create_engine
from src.common import parse_config
from src.database import SequenceInfo
from tqdm import tqdm

IUPAC_CHARS = "ACGTRYSWKMBDHVN"
GAP_CHARS = "-.~_"

TRANS_DICT = {}
for c in IUPAC_CHARS:
    TRANS_DICT[ord(c)] = c
    TRANS_DICT[ord(c.lower())] = c
TRANS_DICT[ord("U")] = "T"
TRANS_DICT[ord("u")] = "T"
for c in GAP_CHARS:
    TRANS_DICT[ord(c)] = "-"

TRANS_DICT.update({i: ord("N") for i in range(256) if i not in TRANS_DICT})


def ambiguity_fraction(bases: np.ndarray) -> float:
    if len(bases) == 0:
        return 1.0

    is_unambiguous = (
        (bases == ord("A")) | (bases == ord("C")) | (bases == ord("G")) | (bases == ord("T"))
    )
    return 1.0 - float(is_unambiguous.mean())


def sparse_sequence(aligned_sequence: str) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.frombuffer(aligned_sequence.encode("ascii"), dtype=np.uint8)
    positions = np.flatnonzero(encoded != ord("-")).astype(np.uint16)
    bases = encoded[positions].copy()
    return positions, bases


@dataclass
class Config:
    input_fasta: Path
    output_database: Path


def main() -> None:
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    num_sequences = 0
    alignment_length = None
    seen_hashes = {}

    sqlite_url = f"sqlite:///{cfg.output_database}"
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

                normalized_seq = str(record.seq).translate(TRANS_DICT)

                positions, bases = sparse_sequence(normalized_seq)

                seq_hash = hashlib.sha256(bases.tobytes()).hexdigest()
                duplicate_of = seen_hashes.get(seq_hash)
                if duplicate_of is None:
                    seen_hashes[seq_hash] = record.id

                tax_fields = record.description.removeprefix(record.id).strip().split(";")

                session.add(
                    SequenceInfo(
                        sequence_id=record.id,
                        tax_phylum=tax_fields[-4],
                        tax_class=tax_fields[-3],
                        tax_order=tax_fields[-2],
                        tax_species=tax_fields[-1],
                        aligned_zlib=zlib.compress(normalized_seq.encode("ascii")),
                        positions_bytes=positions.tobytes(),
                        bases_bytes=bases.tobytes(),
                        length=len(positions),
                        ambiguity_fraction=ambiguity_fraction(bases),
                        first_column=int(positions[0]),
                        last_column=int(positions[-1]),
                        duplicate_of=duplicate_of,
                    )
                )

                if num_sequences % batch_size == 0:
                    session.commit()

            session.commit()


if __name__ == "__main__":
    main()
