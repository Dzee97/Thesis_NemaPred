from typing import cast

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Entrez.Parser import DictionaryElement

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
    input_trait_data: Path
    output_parquet: Path
    worms_cache: Path


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
            if (
                record["valid_name"] is not None
                and record["scientificname"] != record["valid_name"]
            ):
                new_record = query_worms(record["valid_name"])
                record = new_record if new_record is not None else record
            break

    return record


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


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    Entrez.email = "your.email@example.com"

    df_trait_genera = pd.read_csv("data/processed/Trait_data.csv", usecols=["worms_genus"])
    genera = df_trait_genera.worms_genus.to_list()
    all_records = []

    worms_cache = load_worms_cache(cfg.worms_cache)

    for genus in genera:
        searches = {
            "18S": f"{genus}[Organism] AND (18S rRNA OR 18S rDNA OR 18S ribosomal RNA OR small subunit rRNA OR small subunit ribosomal RNA OR SSU)",
            "28S": f"{genus}[Organism] AND (28S rRNA OR 28S rDNA OR 28S ribosomal RNA OR large subunit rRNA OR large subunit ribosomal RNA OR LSU OR D2-D3)",
            "COI": f"{genus}[Organism] AND (COI OR CO1 OR COX1 OR MT-CO1 OR MTCO1 OR cytochrome c oxidase subunit 1 OR cytochrome c oxidase subunit I)",
        }

        for gene, query in searches.items():
            print(f"Checking total hits for {genus} ({gene})...")

            # 1. First, get the total count of matches
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=0)
            search_info = cast(DictionaryElement, Entrez.read(handle))
            handle.close()

            total_count = int(search_info["Count"])
            print(f"Found {total_count} total records for {genus} {gene}.")

            if total_count == 0:
                continue

            # 2. Page through all results in batches of 500
            batch_size = 500
            for start in range(0, total_count, batch_size):
                print(f"Fetching records {start} to {min(start + batch_size, total_count)}...")

                handle = Entrez.esearch(
                    db="nucleotide", term=query, retstart=start, retmax=batch_size
                )
                batch_record = cast(DictionaryElement, Entrez.read(handle))
                handle.close()

                id_list = batch_record["IdList"]
                if not id_list:
                    continue

                # 3. Fetch the actual GenBank flat files for this batch
                fetch_handle = Entrez.efetch(
                    db="nucleotide", id=id_list, rettype="gb", retmode="text"
                )

                for seq_record in SeqIO.parse(fetch_handle, "genbank"):
                    voucher: str | None = None
                    species: str | None = seq_record.annotations.get("organism", None)

                    for feature in seq_record.features:
                        if feature.type == "source":
                            qualifiers = feature.qualifiers
                            source_species: str | None = qualifiers.get("organism", [None])[0]
                            if source_species != species:
                                raise ValueError(
                                    f"Annotation species not equal to source species: {species} vs {source_species}"
                                )
                            voucher = (
                                qualifiers.get("specimen_voucher", [None])[0]
                                or qualifiers.get("isolate", [None])[0]
                                or qualifiers.get("culture_collection", [None])[0]
                                or qualifiers.get("strain", [None])[0]
                            )

                    if species and seq_record.seq.defined:
                        if voucher is None:
                            voucher = seq_record.id

                        species_parts = species.split()
                        species_query = (
                            " ".join(species_parts[:2]) if len(species_parts) > 1 else ""
                        )
                        genus_query = species_parts[0]

                        worms_record = get_worms_taxonomy(
                            [species_query, genus_query, genus], worms_cache
                        )
                        if worms_record is None:
                            continue

                        all_records.append(
                            {
                                "worms_rank": worms_record["rank"],
                                "worms_ismarine": worms_record["isMarine"],
                                "worms_class": worms_record["class"],
                                "worms_order": worms_record["order"],
                                "worms_family": worms_record["family"],
                                "worms_genus": worms_record["genus"],
                                "worms_name": worms_record["scientificname"],
                                "ncbi_species": species,
                                "voucher": voucher,
                                "gene": gene,
                                "accession": seq_record.id,
                                "length": len(seq_record.seq),
                                "sequence": str(seq_record.seq),
                            }
                        )
                fetch_handle.close()

            save_worms_cache(worms_cache, cfg.worms_cache)

    # --- Post-processing and pairing (same as before) ---
    df = pd.DataFrame(all_records)
