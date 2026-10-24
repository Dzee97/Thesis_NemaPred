import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import skbio
from Bio import Entrez, SeqIO
from Bio.Entrez.Parser import DictionaryElement
from src.common import parse_config
from tqdm import tqdm


@dataclass
class Config:
    input_trait_data: Path
    output_parquet: Path
    output_18S_fasta: Path
    output_28S_fasta: Path
    output_COI_template_fasta: Path
    output_COI_fragment_fasta: Path
    worms_cache: Path
    entrez_cache: Path


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
            records = records[:1]
            # raise RuntimeError(f"Multiple appected nematode WoRMS records found for: {query}")

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


def process_entrez_record(seq_record, query_gene, worms_cache, target_seq_type) -> dict | None:
    ncbi_species: str | None = seq_record.annotations.get("organism", None)
    if ncbi_species is None or not seq_record.seq.defined:
        return None

    name_feature = None
    source_feature = None
    cds_feature = None

    cds_codon_start = None
    cds_product = None
    cds_translation = None

    for feature in seq_record.features:
        if feature.type == "source":
            source_feature = feature
        elif feature.type == "rRNA" or feature.type == "gene":
            if name_feature is None:
                name_feature = feature
            else:
                return None
        elif feature.type == "CDS":
            cds_feature = feature

    if name_feature is None:
        return None
    else:
        qualifiers = name_feature.qualifiers
        seq_type = name_feature.type
        if seq_type != target_seq_type:
            return None
        ncbi_gene = (
            qualifiers.get("product", [None])[0]
            if name_feature.type == "rRNA"
            else qualifiers.get("gene", [None])[0]
        )

    if source_feature is None:
        voucher = None
    else:
        qualifiers = source_feature.qualifiers
        source_species: str | None = qualifiers.get("organism", [None])[0]
        if source_species != ncbi_species:
            raise ValueError(
                f"Annotation species not equal to source species: {ncbi_species} vs {source_species}"
            )
        voucher = (
            qualifiers.get("specimen_voucher", [None])[0]
            or qualifiers.get("isolate", [None])[0]
            or qualifiers.get("culture_collection", [None])[0]
            or qualifiers.get("strain", [None])[0]
        )

    if voucher is None:
        voucher = seq_record.id

    if cds_feature is not None:
        cds_gene = cds_feature.qualifiers.get("gene", [None])[0]
        if cds_gene != ncbi_gene:
            raise ValueError(f"Gene feature gene not equal to CDS gene: {ncbi_gene} vs {cds_gene}")
        cds_codon_start = cds_feature.qualifiers.get("codon_start", [None])[0]
        cds_product = cds_feature.qualifiers.get("product", [None])[0]
        cds_translation: str = cds_feature.qualifiers.get("translation", [None])[0]
    elif query_gene == "COI":
        return None

    species_parts = ncbi_species.split()
    species_query = " ".join(species_parts[:2]) if len(species_parts) > 1 else ""
    genus_query = species_parts[0]

    worms_record = get_worms_taxonomy([species_query, genus_query], worms_cache)
    if worms_record is None:
        return

    return {
        "worms_rank": worms_record["rank"],
        "worms_ismarine": bool(worms_record["isMarine"]),
        "worms_class": worms_record["class"],
        "worms_order": worms_record["order"],
        "worms_family": worms_record["family"],
        "worms_genus": worms_record["genus"],
        "worms_name": worms_record["scientificname"],
        "ncbi_species": ncbi_species,
        "voucher": voucher,
        "query_gene": query_gene,
        "ncbi_gene": ncbi_gene,
        "accession": seq_record.id,
        "description": seq_record.description,
        "seq_length": len(seq_record.seq),
        "seq": str(seq_record.seq),
        "cds_codon_start": cds_codon_start,
        "cds_product": cds_product,
        "cds_translation": cds_translation,
        "cds_length": len(cds_translation) if cds_translation is not None else None,
    }


def load_cache(path: Path, name: str) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            cache = pickle.load(f)
            print(f"Loaded {name} cache from disk with {len(cache)} entries")
            return cache

    return {}


def save_cache(cache: dict, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    Entrez.email = "your.email@example.com"

    df_trait_genera = pd.read_csv(
        "data/processed/Trait_data.csv", usecols=["worms_order", "worms_genus"]
    )
    search_taxa = df_trait_genera.worms_order.unique()
    all_records = []

    worms_cache = load_cache(cfg.worms_cache, "WoRMS")
    entrez_cache = load_cache(cfg.entrez_cache, "Entrez")

    queries_18S = [
        "18S rRNA",
        "18S ribosomal RNA",
        "small subunit rRNA",
        "small subunit ribosomal RNA",
    ]
    queries_28S = [
        "28S rRNA",
        "28S ribosomal RNA",
        "large subunit rRNA",
        "large subunit ribosomal RNA",
    ]
    queries_COI = [
        "COI",
        "COX1",
        "cytochrome c oxidase subunit 1",
        "cytochrome c oxidase subunit I",
    ]

    for taxon in search_taxa:
        # Restricting the text match to the title/definition line of the record
        searches = {
            "18S": (
                "rRNA",
                f"{taxon}[Organism] AND ({' OR '.join(f'{q}[Title]' for q in queries_18S)})",
            ),
            "28S": (
                "rRNA",
                f"{taxon}[Organism] AND ({' OR '.join(f'{q}[Title]' for q in queries_28S)})",
            ),
            "COI": (
                "gene",
                f"{taxon}[Organism] AND ({' OR '.join(f'{q}[Title]' for q in queries_COI)})",
            ),
        }

        for gene, (seq_type, query) in searches.items():
            print(f"\nChecking total hits for {taxon} ({gene})...")

            # 1. First, get the total count of matches
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=0)
            search_info = cast(DictionaryElement, Entrez.read(handle))
            handle.close()

            total_count = int(search_info["Count"])
            print(f"Found {total_count} total records for {taxon} {gene}.")

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

                cached_records = []
                fetch_id_list = []
                for id in id_list:
                    if id in entrez_cache:
                        cached_records.append(entrez_cache[id])
                    else:
                        fetch_id_list.append(id)

                print(
                    f"{len(cached_records)} records found in cache, fetching {len(fetch_id_list)}"
                )

                if len(fetch_id_list) > 0:
                    # 3. Fetch the actual GenBank flat files for this batch
                    fetch_handle = Entrez.efetch(
                        db="nucleotide", id=fetch_id_list, rettype="gb", retmode="text"
                    )

                    for id, seq_record in zip(fetch_id_list, SeqIO.parse(fetch_handle, "genbank")):
                        entrez_cache[id] = seq_record
                        processed_record = process_entrez_record(
                            seq_record, gene, worms_cache, seq_type
                        )
                        if processed_record is not None:
                            all_records.append(processed_record)

                    fetch_handle.close()
                    save_cache(entrez_cache, cfg.entrez_cache)

                for seq_record in cached_records:
                    processed_record = process_entrez_record(
                        seq_record, gene, worms_cache, seq_type
                    )
                    if processed_record is not None:
                        all_records.append(processed_record)

                save_cache(worms_cache, cfg.worms_cache)

    # --- Post-processing and pairing (same as before) ---
    pa_table = pa.Table.from_pylist(all_records)
    df: pd.DataFrame = pa_table.to_pandas(types_mapper=pd.ArrowDtype)

    df = df.sort_values(
        [
            "worms_class",
            "worms_order",
            "worms_family",
            "worms_genus",
            "worms_name",
            "ncbi_species",
            "voucher",
            "query_gene",
            "ncbi_gene",
            "accession",
        ]
    )

    # Keep only marine species
    df = df[df.worms_ismarine]
    df = df.set_index("accession")

    trait_genera = df_trait_genera.worms_genus.unique()

    df["trait_genus"] = df.worms_genus.isin(trait_genera)

    df.to_parquet(cfg.output_parquet, engine="pyarrow")

    with (
        open(cfg.output_18S_fasta, "w") as f_18S,
        open(cfg.output_28S_fasta, "w") as f_28S,
        open(cfg.output_COI_template_fasta, "w") as f_COI_template,
        open(cfg.output_COI_fragment_fasta, "w") as f_COI_fragment,
    ):
        for accession, row in df.iterrows():
            seq = row.seq
            if row.query_gene == "18S":
                f = f_18S
            elif row.query_gene == "28S":
                f = f_28S
            else:
                if row.seq_length > 900:
                    f = f_COI_template
                else:
                    f = f_COI_fragment
                seq = row.cds_translation

            f.write(f">{accession}\n{seq}\n")


if __name__ == "__main__":
    main()
