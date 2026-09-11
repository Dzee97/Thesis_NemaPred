import gzip
import hashlib
import heapq
import io
import tarfile
import zlib
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO

if "snakemake" not in globals():
    from snakemake.script import Snakemake
    snakemake: Snakemake = None


IUPAC_CHARS = "ACGTRYSWKMBDHVN"
GAP_CHARS = "-.~_"

# Normalize all expected sequence symbols in one fast str.translate() call.
trans_dict = {}
for c in IUPAC_CHARS:
    trans_dict[ord(c)] = c
    trans_dict[ord(c.lower())] = c
trans_dict[ord("U")] = "T"
trans_dict[ord("u")] = "T"
for c in GAP_CHARS:
    trans_dict[ord(c)] = "-"

NORMALIZE_TRANS = str.maketrans(trans_dict)
NORMALIZE_TRANS.update({i: ord("N") for i in range(256) if i not in trans_dict})


@dataclass
class SequenceInfo:
    """Compact in-memory representation of one normalized aligned sequence."""

    sequence_id: str
    description: str
    aligned_zlib: bytes
    positions: np.ndarray  # non-gap SILVA alignment columns, zero-based
    bases: np.ndarray      # ASCII codes at those positions
    ambiguity_fraction: float

    @property
    def length(self) -> int:
        return len(self.positions)

    @property
    def first_column(self) -> int:
        return int(self.positions[0])

    @property
    def last_column(self) -> int:
        return int(self.positions[-1])

    def aligned_sequence(self) -> str:
        return zlib.decompress(self.aligned_zlib).decode("ascii")

    def ungapped_sequence(self) -> str:
        return self.bases.tobytes().decode("ascii")


def ambiguity_fraction(bases: np.ndarray) -> float:
    """Fraction of ungapped symbols that are not unambiguous A/C/G/T."""
    if len(bases) == 0:
        return 1.0

    is_dna = (
        (bases == ord("A"))
        | (bases == ord("C"))
        | (bases == ord("G"))
        | (bases == ord("T"))
    )
    return 1.0 - float(is_dna.mean())


def sparse_sequence(aligned_sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """Return non-gap alignment positions and their nucleotide symbols."""
    encoded = np.frombuffer(aligned_sequence.encode("ascii"), dtype=np.uint8)
    positions = np.flatnonzero(encoded != ord("-")).astype(np.int32)
    bases = encoded[positions].copy()
    return positions, bases


def is_exactly_contained(short: SequenceInfo, long: SequenceInfo) -> bool:
    """
    Return True when every non-gap symbol in `short` occurs with exactly the
    same symbol at the same SILVA alignment column in `long`.
    """
    if short.length >= long.length:
        return False

    if long.first_column > short.first_column or long.last_column < short.last_column:
        return False

    idx = np.searchsorted(long.positions, short.positions)
    if np.any(idx >= long.length):
        return False

    return (
        np.array_equal(long.positions[idx], short.positions)
        and np.array_equal(long.bases[idx], short.bases)
    )


def find_container(
    sequence_index: int,
    sequences: list[SequenceInfo],
    token_index: dict[tuple[int, int], array],
    num_anchors: int = 3,
) -> int | None:
    """
    Find a retained longer sequence that exactly contains the query.

    Candidate generation uses a few rare alignment-position/base tokens.
    The final decision always uses the complete exact containment test.
    """
    short = sequences[sequence_index]

    # Any true container must contain every token in the shorter sequence.
    # If even one token has never occurred in a retained longer sequence,
    # containment is impossible.
    postings = []
    for pos, base in zip(short.positions, short.bases):
        matches = token_index.get((int(pos), int(base)))
        if matches is None:
            return None
        postings.append(matches)

    # Intersect a few of the rarest tokens to get a small candidate set.
    anchor_lists = heapq.nsmallest(num_anchors, postings, key=len)
    candidates = set(anchor_lists[0])
    for matches in anchor_lists[1:]:
        candidates.intersection_update(matches)
        if not candidates:
            return None

    # Prefer the longest exact container. The index only contains sequences
    # that have already survived containment filtering.
    candidate_order = sorted(
        candidates,
        key=lambda idx: sequences[idx].length,
        reverse=True,
    )

    for candidate_index in candidate_order:
        long = sequences[candidate_index]
        if is_exactly_contained(short, long):
            return candidate_index

    return None


def remove_exact_containment(
    sequences: list[SequenceInfo],
) -> tuple[set[int], dict[int, int]]:
    """
    Remove exact alignment-aware subsequences, keeping longer representatives.

    Sequences are processed longest first. Only retained representatives are
    indexed, so a removed fragment can never become a representative itself.
    """
    order = sorted(
        range(len(sequences)),
        key=lambda idx: (-sequences[idx].length, sequences[idx].sequence_id),
    )

    retained = set()
    contained_by = {}

    # (alignment column, exact symbol) -> retained sequence indices
    token_index = defaultdict(lambda: array("I"))

    for sequence_index in order:
        info = sequences[sequence_index]
        container_index = find_container(sequence_index, sequences, token_index)

        if container_index is not None:
            contained_by[sequence_index] = container_index
            continue

        retained.add(sequence_index)
        for pos, base in zip(info.positions, info.bases):
            token_index[(int(pos), int(base))].append(sequence_index)

    return retained, contained_by


def main() -> None:
    min_ungapped_length = int(snakemake.params.min_ungapped_length)
    max_ambiguity_fraction = float(snakemake.params.max_ambiguity_fraction)

    input_fasta = Path(snakemake.input.input_fasta)
    output_aligned_fasta = Path(snakemake.output.output_aligned_fasta)
    output_ungapped_fasta = Path(snakemake.output.output_ungapped_fasta)
    output_length_plot = Path(snakemake.output.output_length_plot)
    output_column_plot = Path(snakemake.output.output_column_plot)

    # Optional but useful Snakemake output. If it is not configured, the script
    # still works; it simply does not write the mapping table.
    redundancy_output = getattr(snakemake.output, "output_redundancy_tsv", None)
    output_redundancy_tsv = Path(redundancy_output) if redundancy_output else None

    for path in [
        output_aligned_fasta,
        output_ungapped_fasta,
        output_length_plot,
        output_column_plot,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if output_redundancy_tsv is not None:
        output_redundancy_tsv.parent.mkdir(parents=True, exist_ok=True)

    scanned = 0
    dropped_length = 0
    dropped_ambiguity = 0
    dropped_duplicate = 0

    alignment_length = None
    seen_hashes = {}
    sequences = []
    duplicate_rows = []

    try:
        # ------------------------------------------------------------------
        # 1. Normalize, basic QC, and exact aligned-sequence deduplication
        # ------------------------------------------------------------------
        with tarfile.open(input_fasta, "r:gz") as tar:
            member = tar.getmembers()[0]
            tar_file_obj = tar.extractfile(member)
            if tar_file_obj is None:
                raise TypeError(f"Could not extract data stream from: {member.name}")

            with io.TextIOWrapper(tar_file_obj, encoding="utf-8", errors="replace") as in_text:
                for record in SeqIO.parse(in_text, "fasta"):
                    scanned += 1

                    normalized_seq = str(record.seq).translate(NORMALIZE_TRANS)

                    if alignment_length is None:
                        alignment_length = len(normalized_seq)
                    elif len(normalized_seq) != alignment_length:
                        raise ValueError(
                            f"Inconsistent alignment length for {record.id}: "
                            f"{len(normalized_seq)} != {alignment_length}"
                        )

                    positions, bases = sparse_sequence(normalized_seq)

                    if len(positions) < min_ungapped_length:
                        dropped_length += 1
                        continue

                    amb_fraction = ambiguity_fraction(bases)
                    if amb_fraction > max_ambiguity_fraction:
                        dropped_ambiguity += 1
                        continue

                    # Deduplicate in the SILVA coordinate system, not only by
                    # the ungapped nucleotide string.
                    seq_hash = hashlib.sha256(normalized_seq.encode("ascii")).hexdigest()
                    duplicate_of = seen_hashes.get(seq_hash)
                    if duplicate_of is not None:
                        dropped_duplicate += 1
                        duplicate_rows.append(
                            (record.id, duplicate_of, len(positions))
                        )
                        continue

                    seen_hashes[seq_hash] = record.id

                    sequences.append(
                        SequenceInfo(
                            sequence_id=record.id,
                            description=record.description,
                            aligned_zlib=zlib.compress(normalized_seq.encode("ascii"), level=1),
                            positions=positions,
                            bases=bases,
                            ambiguity_fraction=amb_fraction,
                        )
                    )

        if alignment_length is None:
            raise ValueError("No FASTA records were found in the input archive.")

        # ------------------------------------------------------------------
        # 2. Exact alignment-aware containment filtering
        # ------------------------------------------------------------------
        retained, contained_by = remove_exact_containment(sequences)

        retained_indices = sorted(retained)
        id_to_index = {info.sequence_id: idx for idx, info in enumerate(sequences)}

        # Build the redundancy table only after containment filtering so exact
        # duplicates point to the final retained representative as well.
        redundancy_rows = []

        for removed_id, duplicate_of, removed_length in duplicate_rows:
            representative_index = id_to_index[duplicate_of]
            representative_index = contained_by.get(representative_index, representative_index)
            representative = sequences[representative_index]
            redundancy_rows.append(
                (
                    removed_id,
                    representative.sequence_id,
                    removed_length,
                    representative.length,
                    "exact_aligned_duplicate",
                )
            )

        for removed_index, representative_index in contained_by.items():
            removed = sequences[removed_index]
            representative = sequences[representative_index]
            redundancy_rows.append(
                (
                    removed.sequence_id,
                    representative.sequence_id,
                    removed.length,
                    representative.length,
                    "exact_alignment_containment",
                )
            )

        # ------------------------------------------------------------------
        # 3. Write final FASTA files and collect diagnostics
        # ------------------------------------------------------------------
        ungapped_lengths = []
        column_occupancy = np.zeros(alignment_length, dtype=np.uint32)

        with gzip.open(output_aligned_fasta, "wt", encoding="utf-8") as out_aligned, \
             gzip.open(output_ungapped_fasta, "wt", encoding="utf-8") as out_ungapped:

            for idx in retained_indices:
                info = sequences[idx]
                aligned_seq = info.aligned_sequence()
                ungapped_seq = info.ungapped_sequence()
                header = f">{info.description}\n"

                out_aligned.write(header)
                out_aligned.write(aligned_seq)
                out_aligned.write("\n")

                out_ungapped.write(header)
                out_ungapped.write(ungapped_seq)
                out_ungapped.write("\n")

                ungapped_lengths.append(info.length)
                column_occupancy[info.positions] += 1

        if output_redundancy_tsv is not None:
            with output_redundancy_tsv.open("w", encoding="utf-8") as out:
                out.write("removed_id\tretained_id\tremoved_length\tretained_length\treason\n")
                for row in redundancy_rows:
                    out.write("\t".join(map(str, row)) + "\n")

        # ------------------------------------------------------------------
        # 4. Diagnostic plots
        # ------------------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.hist(ungapped_lengths, bins=50, edgecolor="black", alpha=0.7)
        plt.title("Distribution of Retained Sequence Lengths (Ungapped)")
        plt.xlabel("Sequence Length (bp)")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_length_plot, dpi=300)
        plt.close()

        plt.figure(figsize=(12, 5))
        x = np.arange(1, alignment_length + 1)
        plt.plot(x, column_occupancy, linewidth=1.2)
        plt.fill_between(x, column_occupancy, alpha=0.25)
        plt.title("Alignment Column Occupancy Profile")
        plt.xlabel("Alignment Position Coordinate")
        plt.ylabel("Number of Retained Sequences")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_column_plot, dpi=300)
        plt.close()

        # ------------------------------------------------------------------
        # 5. Summary
        # ------------------------------------------------------------------
        dropped_containment = len(contained_by)
        kept = len(retained_indices)

        print("\n=== Filtering Summary ===")
        print(f"Total Scanned:               {scanned:,}")
        print(f"Dropped (Too Short):         {dropped_length:,}")
        print(f"Dropped (Ambiguity):         {dropped_ambiguity:,}")
        print(f"Dropped (Aligned Duplicate): {dropped_duplicate:,}")
        print(f"Dropped (Contained):         {dropped_containment:,}")
        print(f"Total Retained:              {kept:,}")
        print("=========================")

        print(f"\nAlignment length: {alignment_length:,} columns")
        print(f"Generated figures:\n - {output_length_plot}\n - {output_column_plot}")
        if output_redundancy_tsv is not None:
            print(f"Redundancy mapping:\n - {output_redundancy_tsv}")

    except Exception:
        output_aligned_fasta.unlink(missing_ok=True)
        output_ungapped_fasta.unlink(missing_ok=True)
        output_length_plot.unlink(missing_ok=True)
        output_column_plot.unlink(missing_ok=True)
        if output_redundancy_tsv is not None:
            output_redundancy_tsv.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()