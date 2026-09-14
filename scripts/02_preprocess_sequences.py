import hashlib
import heapq
import io
import tarfile
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import SeqIO
from sqlmodel import Session, SQLModel, create_engine
from src.common import parse_config
from src.database import SequenceInfo
from tqdm import tqdm

DNA_CHARS = "ACGTN"
GAP_CHARS = "-.~_"

TRANS_DICT = {}
for c in DNA_CHARS:
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

    is_ambiguous = bases == ord("N")
    return float(is_ambiguous.mean())


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

    alignment_length = None
    sequences: list[SequenceInfo] = []

    sqlite_url = f"sqlite:///{cfg.output_database}"
    engine = create_engine(sqlite_url)
    SQLModel.metadata.create_all(engine)

    with tarfile.open(cfg.input_fasta, "r:gz") as tar:
        member = tar.getmembers()[0]
        tar_file = tar.extractfile(member)
        if tar_file is None:
            raise TypeError(f"Could not extract data from: {member.name}")

        with io.TextIOWrapper(tar_file, encoding="utf-8", errors="replace") as text_in:
            for record in tqdm(SeqIO.parse(text_in, "fasta"), desc="Loading sequences"):
                if alignment_length is None:
                    alignment_length = len(record.seq)
                elif len(record.seq) != alignment_length:
                    raise ValueError(
                        f"Inconsistent alignment length for {record.id}: "
                        f"{len(record.seq)} != {alignment_length}"
                    )

                normalized_seq = str(record.seq).translate(TRANS_DICT)

                positions, bases = sparse_sequence(normalized_seq)

                tax_fields = record.description.removeprefix(record.id).strip().split(";")

                sequences.append(
                    SequenceInfo(
                        sequence_id=record.id,
                        tax_phylum=tax_fields[-4],
                        tax_class=tax_fields[-3],
                        tax_order=tax_fields[-2],
                        tax_species=tax_fields[-1],
                        positions_bytes=positions.tobytes(),
                        bases_bytes=bases.tobytes(),
                        length=len(positions),
                        ambiguity_fraction=ambiguity_fraction(bases),
                        first_column=int(positions[0]),
                        last_column=int(positions[-1]),
                    )
                )

    if alignment_length is None:
        raise ValueError("No FASTA records were found.")

    # -------------------------------------------
    # 2. Exact alignment containement filtering
    # -------------------------------------------

    # Collect aligned sequence hashed for detecting exact duplicates
    seen_hashes: dict[str, int] = {}

    # (alignment position, base) -> uncontained sequence indices
    token_index: defaultdict[tuple[int, int], array[int]] = defaultdict(lambda: array("I"))

    # Sequences are processed longest first.
    # That way containment is only checked against previously proven uncontained sequences.
    seq_length_order = sorted(
        range(len(sequences)), key=lambda idx: (-sequences[idx].length, sequences[idx].sequence_id)
    )
    for seq_idx in tqdm(seq_length_order, desc="Determining containment"):
        short_seq = sequences[seq_idx]

        # First check if the sequence is identical to a previous unontained sequence
        # If so, duplication is noted, containment copied from the duplicate and searching stops.
        hasher = hashlib.sha256()
        hasher.update(short_seq.positions_bytes)
        hasher.update(short_seq.bases_bytes)

        seq_hash = hasher.hexdigest()
        dup_idx = seen_hashes.get(seq_hash)
        if dup_idx is None:
            seen_hashes[seq_hash] = seq_idx
        else:
            dup_seq = sequences[dup_idx]
            short_seq.duplicate_of = dup_seq.sequence_id
            short_seq.contained_by = dup_seq.contained_by
            continue

        # Sequences start uncontained until proven otherwise.
        contained = False

        # Loop over every (alignment position, base) pair to find matches in the uncontained sequences.
        # If only one pair has no matches, containment is impossible and searching stops.
        pos_matches: list[array[int]] = []
        for pos, base in zip(short_seq.positions, short_seq.bases):
            matches = token_index.get((int(pos), int(base)))
            if matches is None:
                break
            pos_matches.append(matches)
        else:
            # When we have matches for all positions,
            # intersect the 3 columns with the fewest matches to get a small candidate set.
            rarest_matches = heapq.nsmallest(3, pos_matches, key=len)
            candidates = set(rarest_matches[0])
            for matches in rarest_matches[1:]:
                candidates.intersection_update(matches)

            # Containment if only possible when at least 1 candidate appeared in all 3 columns.
            if candidates:
                # Candidate containers are processed longest first
                candidate_order = sorted(candidates, key=lambda idx: -sequences[idx].length)
                for candidate_idx in candidate_order:
                    long_seq = sequences[candidate_idx]

                    # Containment is only possible when the sequence sits completely
                    # within the candidate alignment coordinates.
                    if (
                        short_seq.first_column < long_seq.first_column
                        or short_seq.last_column > long_seq.last_column
                    ):
                        continue

                    # Find the index in the candidate alignment positions where the short sequence should start.
                    start_idx = np.searchsorted(long_seq.positions, short_seq.first_column)

                    # The sequence is contained if both the alignment positions and bases are equal to the candidate
                    # after slicing from the starting index up until the sequence length.
                    contained = np.array_equal(
                        short_seq.positions,
                        long_seq.positions[start_idx : start_idx + short_seq.length],
                    ) and np.array_equal(
                        short_seq.bases, long_seq.bases[start_idx : start_idx + short_seq.length]
                    )

                    # Save the containment reference in the databse record
                    if contained:
                        short_seq.contained_by = long_seq.sequence_id
                        break

        # When proven uncontained, add the (alignment position, base) information to the token index.
        if not contained:
            for pos, base in zip(short_seq.positions, short_seq.bases):
                token_index[(int(pos), int(base))].append(seq_idx)

    # -------------------------------------------
    # 3. Save database
    # -------------------------------------------

    with Session(engine) as session:
        session.add_all(sequences)
        session.commit()


if __name__ == "__main__":
    main()
