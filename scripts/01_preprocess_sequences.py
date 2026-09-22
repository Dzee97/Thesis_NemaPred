from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from src.common import parse_config
from src.sequenceinfo import SequenceInfo
from tqdm import tqdm


@dataclass
class Config:
    input_fasta: Path
    input_uc: Path
    output_parquet: Path
    header_type: str


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
    # 2. Load clustering results
    # ---------------------------------------

    uc_headers = [
        "record_type",
        "cluster_id",
        "length",
        "pct_identity",
        "strand",
        "query_start",
        "target_start",
        "alignment",
        "query_label",
        "target_label",
    ]

    df_uc = pd.read_csv(cfg.input_uc, sep="\t", comment="#", names=uc_headers, na_values="*")
    df_uc = df_uc[df_uc["record_type"] != "C"]
    df_uc.set_index("query_label", inplace=True)

    df_uc["target_length"] = df_uc["target_label"].map(df_uc["length"])

    exact_match = df_uc["pct_identity"] == 100.0
    df_uc["duplicate"] = exact_match & (df_uc["length"] == df_uc["target_length"])
    df_uc["contained"] = exact_match & (df_uc["length"] != df_uc["target_length"])
    df_uc["centroid"] = df_uc["record_type"] == "S"

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

        uc = df_uc.loc[accession]

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
                cluster_centroid=uc["centroid"],
                cluster_id=uc["cluster_id"],
                cluster_pct_identity=uc["pct_identity"],
                cluster_target=uc["target_label"],
                cluster_duplicate=uc["duplicate"],
                cluster_contained=uc["contained"],
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
    # 4. Save to compressed dataframe
    # -------------------------------------------

    df = pd.DataFrame(sequences)
    df.to_parquet(cfg.output_parquet)


if __name__ == "__main__":
    main()
