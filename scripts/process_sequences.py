import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import skbio
from src.common import parse_config
from tqdm import tqdm


@dataclass
class Config:
    input_ref_fasta: Path
    input_ref_uc: Path
    input_otu_fasta: Path
    input_out_fasta: Path
    output_parquet: Path
    worms_cache: Path


@dataclass
class SequenceInfo:
    accession: str
    description: str
    length: int
    ambiguity_frac: float
    rna_seq: np.ndarray
    msa_pos: np.ndarray
    first_pos: int
    last_pos: int
    seq_type: Literal["ref", "outgroup", "otu"]
    worms_aphiaid: int | None = None
    worms_rank: str | None = None
    worms_ismarine: bool | None = None
    worms_name: str | None = None
    worms_valid_name: str | None = None
    worms_valid_genus: str | None = None


def get_worms_taxonomy(queries: list[str], worms_cache: dict) -> dict[str, str] | None:
    worms_url = "https://www.marinespecies.org/rest"
    params = {"like": "false", "marine_only": "false"}

    def query_worms(query: str):
        if query in worms_cache:
            return worms_cache[query]

        url = f"{worms_url}/AphiaRecordsByName/{quote(query)}"

        response = requests.get(url, params=params)
        if response.status_code in [404, 204]:
            worms_cache[query] = None
            return None

        records: list[dict] = response.json()
        if len(records) > 1:
            records = [r for r in records if r["status"] == "accepted"]
        if len(records) > 1:
            records = [r for r in records if r["rank"] in ["Genus", "Species"]]
        if len(records) > 1:
            records = [r for r in records if r["phylum"] == "Nematoda"]
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


def process_record(
    seq: skbio.RNA,
    alignment_length: int,
    seq_type: Literal["ref", "outgroup", "otu"],
    worms_cache: dict | None = None,
) -> SequenceInfo:
    if len(seq) != alignment_length:
        raise ValueError(
            f"Inconsistent alignment length for {seq.metadata['id']}: "
            f"{len(seq)} != {alignment_length}"
        )

    non_gap_mask = ~seq.gaps()
    ungapped_seq = seq.degap().values.view(np.uint8)
    positions = np.flatnonzero(non_gap_mask).astype(np.uint16)

    length = len(ungapped_seq)
    ambiguity_frac = seq.degenerates().sum() / length

    seq_info = SequenceInfo(
        accession=seq.metadata["id"],
        description=seq.metadata["description"],
        length=length,
        ambiguity_frac=ambiguity_frac,
        rna_seq=ungapped_seq,
        msa_pos=positions,
        first_pos=positions[0],
        last_pos=positions[-1],
        seq_type=seq_type,
    )

    if seq_type == "otu":
        return seq_info

    if worms_cache is None:
        worms_cache = {}

    hyp_genus, hyp_species = seq.metadata["description"].split(";")
    worms_queries = [hyp_species, hyp_genus]
    worms_record = get_worms_taxonomy(worms_queries, worms_cache)
    if worms_record is None:
        return seq_info
        # raise RuntimeError(f"No WoRMS record found for: {seq.metadata['description']}")

    seq_info.worms_aphiaid = int(worms_record["AphiaID"])
    seq_info.worms_rank = worms_record["rank"]
    seq_info.worms_ismarine = bool(worms_record["isMarine"])
    seq_info.worms_name = worms_record["scientificname"]
    seq_info.worms_valid_name = worms_record["valid_name"]
    seq_info.worms_valid_genus = (
        seq_info.worms_valid_name.split()[0]
        if seq_info.worms_valid_name and seq_info.worms_rank in ["Genus", "Species"]
        else None
    )

    return seq_info


def load_worms_cache(path: Path) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            worms_cache = pickle.load(f)
            print(f"Loaded WoRMS cache from disk with {len(worms_cache)} entries")
            return worms_cache

    return {}


def save_worms_cache(worms_cache: dict, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(worms_cache, f)


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    # Intermediate structure for saving record data
    sequence_rows = []
    alignment_length = 50000

    # Try to load exsisting worms cache from disk
    worms_cache = load_worms_cache(cfg.worms_cache)

    # Read each reference sequence, saving the sparse alignment representation
    seq: skbio.RNA
    for count, seq in tqdm(
        enumerate(skbio.read(cfg.input_ref_fasta, format="fasta", constructor=skbio.RNA)),
        desc="Processing reference sequences",
    ):
        sequence_rows.append(
            process_record(
                seq=seq,
                alignment_length=alignment_length,
                seq_type="ref",
                worms_cache=worms_cache,
            )
        )

        if count % 1000 == 0:
            save_worms_cache(worms_cache, cfg.worms_cache)

    # Now process the outgroup sequences
    seq: skbio.RNA
    for seq in tqdm(
        skbio.read(cfg.input_out_fasta, format="fasta", constructor=skbio.RNA),
        desc="Processing outgroup sequences",
    ):
        sequence_rows.append(
            process_record(
                seq=seq,
                alignment_length=alignment_length,
                seq_type="outgroup",
                worms_cache=worms_cache,
            )
        )

    save_worms_cache(worms_cache, cfg.worms_cache)

    # Finally process the OTU sequences
    seq: skbio.RNA
    for seq in tqdm(
        skbio.read(cfg.input_otu_fasta, format="fasta", constructor=skbio.RNA),
        desc="Processing OTU sequences",
    ):
        sequence_rows.append(
            process_record(seq=seq, alignment_length=alignment_length, seq_type="otu")
        )

    # Save to a PyArrow backed Pandas Dataframe
    pa_table = pa.Table.from_pylist([s.__dict__ for s in sequence_rows])
    df_seq: pd.DataFrame = pa_table.to_pandas(types_mapper=pd.ArrowDtype)

    df_seq.set_index("accession", inplace=True)

    # Load the VSEARCH clustering results in a PyArrow backed Pandas Dataframe
    print("Merging VSEARCH clustering results")

    uc_headers = [
        "record_type",
        "clust_id",
        "length",
        "clust_pct_identity",
        "strand",
        "query_start",
        "target_start",
        "alignment",
        "query_label",
        "clust_target",
    ]
    df_uc = pd.read_csv(
        cfg.input_ref_uc,
        sep="\t",
        names=uc_headers,
        na_values="*",
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    df_uc = df_uc[df_uc.record_type != "C"]
    df_uc.set_index("query_label", inplace=True)

    df_uc["target_length"] = df_uc["clust_target"].map(df_uc["length"])
    exact_match = df_uc["clust_pct_identity"] == 100.0
    df_uc["clust_duplicate"] = exact_match & (df_uc["length"] == df_uc["target_length"]).fillna(
        False
    )
    df_uc["clust_contained"] = exact_match & (df_uc["length"] != df_uc["target_length"]).fillna(
        False
    )
    df_uc["clust_centroid"] = df_uc["record_type"] == "S"

    keep_headers = [
        "clust_id",
        "clust_pct_identity",
        "clust_target",
        "clust_duplicate",
        "clust_contained",
        "clust_centroid",
    ]
    df_uc = df_uc[keep_headers]

    # Join both Dataframes and save to a compact Parquet file
    df = df_seq.join(df_uc)
    df["clust_target_genus"] = df["clust_target"].map(df["worms_valid_genus"])
    df["clust_genus_agree"] = df["worms_valid_genus"] == df["clust_target_genus"]

    df.to_parquet(cfg.output_parquet, engine="pyarrow", compression="snappy")


if __name__ == "__main__":
    main()
