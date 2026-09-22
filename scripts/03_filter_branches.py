import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from src.common import parse_config
from src.sequenceinfo import SequenceInfo
from tqdm import tqdm


@dataclass
class Config:
    input_parquet: Path
    input_fast_trees: Path
    cutoff_quantile: float
    output_fasta: Path


def main():
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    tip_lengths = defaultdict(list)
    pattern = re.compile(r"([\w.-]+)_\w+:([0-9\.eE+-]+)(?=[,\)])")

    with open(cfg.input_fast_trees, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            matches = pattern.findall(line)
            for accession, length in matches:
                tip_lengths[accession].append(float(length))

    avg_tip_length = {
        accession: (sum(lengths) / len(lengths)) for accession, lengths in tip_lengths.items()
    }

    df = pd.read_parquet(cfg.input_parquet)
    df["avg_tip_length"] = df["accession"].map(avg_tip_length)

    df = df[df["avg_tip_length"] < df["avg_tip_length"].quantile(cfg.cutoff_quantile)]

    with open(cfg.output_fasta, "w") as f:
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Writing filtered sequences to FASTA"
        ):
            seq = SequenceInfo(**row)
            f.write(f">{seq.fasta_header}\n{seq.aligned_sequence}\n")


if __name__ == "__main__":
    main()
