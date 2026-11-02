import argparse
import pickle
from dataclasses import fields, is_dataclass
from pathlib import Path
from urllib.parse import quote

import requests
from snakemake.script import Snakemake


def parse_config[T](config_cls: type[T], snakemake: Snakemake | None) -> T:
    if not is_dataclass(config_cls):
        raise TypeError("Expected a dataclass config definition")

    if snakemake is not None:
        cfg = config_cls(
            **(dict(snakemake.input) | dict(snakemake.output) | dict(snakemake.params))
        )
    else:
        print("Snakemake not in globals")
        parser = argparse.ArgumentParser()
        for field in fields(config_cls):
            flag = f"--{field.name}"
            kwargs = {"type": field.type, "required": True}
            if field.type is bool:
                kwargs = {"action": "store_true", "required": False}

            parser.add_argument(flag, **kwargs)

        cfg = config_cls(**vars(parser.parse_args()))

    return cfg


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
