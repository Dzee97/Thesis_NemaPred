from dataclasses import dataclass
from pathlib import Path
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
    input_fasta: Path
    input_uc: Path
    input_outgroup: Path
    output_parquet: Path


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
    cols: dict[str, list],
    silva_header: bool,
    is_outgroup: bool,
    worms_cache: dict,
) -> None:
    if len(seq) != alignment_length:
        raise ValueError(
            f"Inconsistent alignment length for {seq.metadata['id']}: "
            f"{len(seq)} != {alignment_length}"
        )

    cols["accession"].append(seq.metadata["id"])
    cols["description"].append(seq.metadata["description"])

    seq_bytes: np.ndarray = seq.values
    non_gap_mask = (seq_bytes != b"-") & (seq_bytes != b".")
    ungapped_seq = seq_bytes[non_gap_mask].view(np.uint8)
    cols["rna_seq"].append(ungapped_seq)

    positions = np.flatnonzero(non_gap_mask).astype(np.uint16)
    cols["msa_pos"].append(positions)

    cols["is_outgroup"].append(is_outgroup)

    if silva_header:
        fields = seq.metadata["description"].split(";")[-1].split()
    else:
        fields = seq.metadata["description"].split()
    hyp_genus = fields[0]
    hyp_species = " ".join(fields[:2]) if len(fields) > 1 else None

    worms_queries = [q for q in [hyp_species, hyp_genus] if q is not None]
    worms_record = get_worms_taxonomy(worms_queries, worms_cache)
    if worms_record is None:
        raise RuntimeError(f"No WoRMS record found for: {seq.metadata['description']}")

    cols["worms_aphiaid"].append(int(worms_record["AphiaID"]))
    cols["worms_rank"].append(worms_record["rank"])
    cols["worms_ismarine"].append(bool(worms_record["isMarine"]))
    cols["worms_name"].append(worms_record["scientificname"])
    cols["worms_valid_name"].append(worms_record["valid_name"])
    cols["worms_valid_genus"].append(worms_record["valid_name"].split()[0])


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    # Intermediate structure for saving record data
    cols = {
        "accession": [],
        "description": [],
        "rna_seq": [],
        "msa_pos": [],
        "is_outgroup": [],
        "worms_aphiaid": [],
        "worms_rank": [],
        "worms_ismarine": [],
        "worms_name": [],
        "worms_valid_name": [],
        "worms_valid_genus": [],
    }
    worms_cache = {}
    alignment_length = 50000

    # First process the outgroup sequences
    seq: skbio.RNA
    for seq in skbio.read(cfg.input_outgroup, format="fasta", constructor=skbio.RNA):
        process_record(
            seq=seq,
            cols=cols,
            alignment_length=alignment_length,
            silva_header=True,
            is_outgroup=True,
            worms_cache=worms_cache,
        )

    # Read each sequence, saving the sparse alignment representation
    seq: skbio.RNA
    for seq in tqdm(
        skbio.read(cfg.input_fasta, format="fasta", constructor=skbio.RNA), desc="Loading sequences"
    ):
        process_record(
            seq=seq,
            cols=cols,
            alignment_length=alignment_length,
            silva_header=False,
            is_outgroup=False,
            worms_cache=worms_cache,
        )

    # Save to a PyArrow backed Pandas Dataframe
    pa_table = pa.Table.from_pydict(cols)
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
        cfg.input_uc,
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
    df_uc["clust_duplicate"] = exact_match & (df_uc["length"] == df_uc["target_length"])
    df_uc["clust_contained"] = exact_match & (df_uc["length"] != df_uc["target_length"])
    df_uc["clust_centroid"] = df_uc["record_type"] == "S"

    keep_headers = [
        "length",
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
    df.to_parquet(cfg.output_parquet, engine="pyarrow", compression="snappy")


if __name__ == "__main__":
    main()
