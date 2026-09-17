import hashlib
import heapq
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from src.common import parse_config
from tqdm import tqdm


@dataclass
class Config:
    input_fasta: Path
    output_parquet: Path
    header_type: Literal["ncbi", "silva"]


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
        return f"{self.accession} {self.description}"

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


def sequence_nomalization_dict() -> dict[int, int]:
    iupac_chars = "ACGTRYSWKMBDHVN"
    gap_chars = "-.~_"

    trans_dict: dict[int, int] = {}
    for c in iupac_chars:
        trans_dict[ord(c)] = ord(c)
        trans_dict[ord(c.lower())] = ord(c)
    trans_dict[ord("U")] = ord("T")
    trans_dict[ord("u")] = ord("T")
    for c in gap_chars:
        trans_dict[ord(c)] = ord("-")

    trans_dict.update({i: ord("N") for i in range(256) if i not in trans_dict})

    return trans_dict


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


def get_worms_taxonomy(queries: list[str], worms_cache: dict) -> dict[str, str] | None:
    worms_url = "https://www.marinespecies.org/rest"
    params = {"like": "false", "marine_only": "false"}

    def query_worms(query: str):
        if query in worms_cache:
            return worms_cache[query]

        url = f"{worms_url}/AphiaRecordsByName/{quote(query)}"

        response = requests.get(url, params=params)
        if response.status_code in [404, 204]:
            return None

        records: list[dict] = response.json()
        records = [r for r in records if r["phylum"] == "Nematoda"]
        if len(records) > 1:
            records = [r for r in records if r["status"] == "accepted"]
        if len(records) > 1:
            records = [r for r in records if r["rank"] in ["Genus", "Species"]]
        if len(records) > 1:
            raise RuntimeError(f"Multiple appected nematode WoRMS records found for: {query}")

        record = records[0] if records else None
        worms_cache[query] = record

        return record

    record = None
    for query in queries:
        record = query_worms(query)
        if record is not None:
            break

    return record


def main() -> None:
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    # ---------------------------------------
    # 1. Load sequences into sparse objects
    # ---------------------------------------
    alignment_length = None
    sequences: list[SequenceInfo] = []
    trans_dict = sequence_nomalization_dict()

    for record in tqdm(SeqIO.parse(cfg.input_fasta, "fasta"), desc="Loading sequences"):
        if alignment_length is None:
            alignment_length = len(record.seq)
        elif len(record.seq) != alignment_length:
            raise ValueError(
                f"Inconsistent alignment length for {record.id}: "
                f"{len(record.seq)} != {alignment_length}"
            )

        normalized_seq = str(record.seq).translate(trans_dict)
        positions, bases = sparse_sequence(normalized_seq)

        accession = record.id
        description = record.description.removeprefix(accession).strip()

        sequences.append(
            SequenceInfo(
                accession=accession,
                description=description,
                positions_bytes=positions.tobytes(),
                bases_bytes=bases.tobytes(),
                length=len(positions),
                ambiguity_fraction=ambiguity_fraction(bases),
                first_column=int(positions[0]),
                last_column=int(positions[-1]),
            )
        )

    # -------------------------------------------
    # 2. WoRMS taxonomic reconciliation
    # -------------------------------------------
    worms_cache = {}

    for seq in tqdm(sequences, desc="WoRMS taxonomic reconciliation"):
        if cfg.header_type == "ncbi":
            desc_fields = seq.description.split()
            hyp_genus = desc_fields[0]
            hyp_species = f"{desc_fields[0]} {desc_fields[1]}" if len(desc_fields) > 1 else None
        elif cfg.header_type == "silva":
            desc_fields = seq.description.split(";")
            species_words = desc_fields[-1].split()
            hyp_genus = species_words[0]
            hyp_species = (
                f"{species_words[0]} {species_words[1]}" if len(species_words) > 1 else None
            )
        worms_queries = [q for q in [hyp_species, hyp_genus] if q is not None]

        worms_record = get_worms_taxonomy(worms_queries, worms_cache)
        if worms_record is None:
            continue

        seq.worms_aphiaid = worms_record["AphiaID"]
        seq.worms_rank = worms_record["rank"]
        seq.worms_name = worms_record["scientificname"]
        seq.worms_genus = (
            seq.worms_name.split()[0] if seq.worms_rank in ["Genus", "Species"] else None
        )
        seq.worms_ismarine = bool(worms_record["isMarine"])
        seq.worms_valid_name = worms_record["valid_name"]
        seq.worms_valid_genus = (
            seq.worms_valid_name.split()[0]
            if seq.worms_valid_name is not None and seq.worms_rank in ["Genus", "Species"]
            else None
        )

    # -------------------------------------------
    # 3. Exact alignment containement filtering
    # -------------------------------------------

    # Collect aligned sequence hashed for detecting exact duplicates
    seen_hashes: dict[str, int] = {}

    # (alignment position, base) -> uncontained sequence indices
    token_index: defaultdict[tuple[int, int], array[int]] = defaultdict(lambda: array("I"))

    # Sequences are processed longest first.
    # That way containment is only checked against previously proven uncontained sequences.
    seq_length_order = sorted(
        range(len(sequences)), key=lambda idx: (-sequences[idx].length, sequences[idx].accession)
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
            short_seq.duplicate_of = dup_seq.accession
            short_seq.contained_by = (
                dup_seq.contained_by if dup_seq.contained_by is not None else dup_seq.accession
            )
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
                        short_seq.contained_by = long_seq.accession
                        break

        # When proven uncontained, add the (alignment position, base) information to the token index.
        if not contained:
            for pos, base in zip(short_seq.positions, short_seq.bases):
                token_index[(int(pos), int(base))].append(seq_idx)

    # -------------------------------------------
    # 4. Save to compressed dataframe
    # -------------------------------------------

    df = pd.DataFrame(sequences)
    df.to_parquet(cfg.output_parquet)


if __name__ == "__main__":
    main()
