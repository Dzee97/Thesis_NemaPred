from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

INPUT_FASTA = Path(
    "data/raw/Project_data_Nematode_traits/18S_references/MarNemaFunDiv_18S_sequences_combined.fasta"
)
INPUT_GENERA = Path("data/raw/Project_data_Nematode_traits/MarNemaFunDiv_Genera.xlsx")

WORMS_URL = "https://www.marinespecies.org/rest"

WORMS_CACHE = {}


def get_worms_taxonomy(description: str):
    description_parts = description.split()

    params = {"like": "false", "marine_only": "true"}

    def query_worms(query: str, params: dict[str, str] = params):
        if query in WORMS_CACHE:
            return WORMS_CACHE[query]

        url = f"{WORMS_URL}/AphiaRecordsByName/{quote(query)}"

        try:
            response = requests.get(url, params=params)
            if response.status_code == 204:
                raise requests.exceptions.HTTPError(
                    "204 No Content: Server returned an empty response.", response=response
                )
            response.raise_for_status()

            records = response.json()
            records = [r for r in records if r["phylum"] == "Nematoda"]
            if len(records) > 1:
                records = [r for r in records if r["status"] == "accepted"]
            if len(records) > 1:
                raise RuntimeError(f"Multiple appected nematode WoRMS records found for: {query}")

            record = records[0]

        except requests.exceptions.HTTPError:
            record = None

        WORMS_CACHE[query] = record

        return record

    # First assume that the first two description words are genus and species
    species = " ".join(description_parts[:2])
    # If that fails, query with only the first word (which is probably genus)
    genus = description_parts[0]

    record = None
    for query in [species, genus]:
        record = query_worms(query)
        if record is not None:
            break

    if record is None:
        raise RuntimeError(f"No WoRMS record found for: {description}")

    return record


def main():
    records = []
    for seq_record in tqdm(SeqIO.parse(INPUT_FASTA, "fasta"), desc="Loading sequences"):
        seq_id = seq_record.id
        seq_description = seq_record.description.removeprefix(seq_id).strip()

        worms_record = get_worms_taxonomy(seq_description)

        record = {
            "worms_scientificname": worms_record["scientificname"],
            "worms_valid_name": worms_record["valid_name"],
            "worms_rank": worms_record["rank"],
        }

        record["valid_genus"] = (
            record["worms_valid_name"].split()[0]
            if record["worms_rank"] in ["Genus", "Species"]
            else None
        )

        record["fasta_genus"] = (
            record["worms_scientificname"].split()[0]
            if record["worms_rank"] in ["Genus", "Species"]
            else None
        )

        record["fasta_id"] = seq_id
        record["fasta_description"] = seq_description
        record["fasta_sequence"] = str(seq_record.seq)

        records.append(record)

    df_fasta = pd.DataFrame.from_records(records)

    df_genera = pd.read_excel(INPUT_GENERA)

    df_fasta["valid_genus_in_traits"] = df_fasta["valid_genus"].isin(df_genera["Genus"])
    df_fasta["fasta_genus_in_traits"] = df_fasta["fasta_genus"].isin(df_genera["Genus"])

    valid_choice = df_fasta["valid_genus"].where(df_fasta["valid_genus_in_traits"])
    fasta_choice = df_fasta["fasta_genus"].where(df_fasta["fasta_genus_in_traits"])
    df_fasta["canonical_genus"] = valid_choice.fillna(fasta_choice)

    canonical_counts = df_fasta["canonical_genus"].value_counts()

    df_genera["count_in_fasta"] = df_genera["Genus"].map(canonical_counts).fillna(0).astype(int)
    df_genera["missing_in_fasta"] = df_genera["count_in_fasta"] == 0

    df_fasta.to_csv("df_fasta.csv")
    df_genera.to_csv("df_genera.csv")

    breakpoint()


if __name__ == "__main__":
    main()
