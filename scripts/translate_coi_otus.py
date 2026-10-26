from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import skbio
from src.common import parse_config


@dataclass
class Config:
    input_fasta: Path
    input_blastx_results: Path
    output_protein_fasta: Path
    output_trimmed_fasta: Path
    min_query_coverage: float
    frame_ambiguity_fraction: float


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    df_blastx = pd.read_csv(
        cfg.input_blastx_results,
        sep="\t",
        names=[
            "seq_id",
            "ref_id",
            "frame",
            "start",
            "end",
            "length",
            "identity",
            "e-value",
            "bitscore",
        ],
    )

    seq_best_frame = df_blastx.groupby("seq_id", sort=False).frame.transform("first")
    df_blastx["best_frame_agree"] = df_blastx.frame == seq_best_frame
    df_blastx["frame_ambiguity_frac"] = df_blastx.groupby(
        "seq_id", sort=False
    ).best_frame_agree.transform("mean")

    df_best = df_blastx.groupby("seq_id", sort=False).head(1)
    df_best = df_best.set_index("seq_id")

    coverage_list = []
    trimmed_seq_list = []
    protein_list = []
    internal_stop_list = []

    seq: skbio.DNA
    for seq in skbio.read(cfg.input_fasta, format="fasta", constructor=skbio.DNA):
        seq_id = seq.metadata["id"]
        if seq_id not in df_best.index:
            continue

        row = df_best.loc[seq.metadata["id"]]
        coverage_list.append((row.end - row.start + 1) / len(seq))

        trimmed_seq = seq[row.frame - 1 :]
        trimmed_seq = trimmed_seq[: len(trimmed_seq) // 3 * 3]
        trimmed_seq_list.append(trimmed_seq)

        protein = trimmed_seq.translate(genetic_code=5)
        internal_stop = protein[:-1].has_stops()

        protein_list.append(str(protein))
        internal_stop_list.append(internal_stop)

    df_best["trimmed_seq"] = trimmed_seq_list
    df_best["coverage"] = coverage_list
    df_best["protein"] = protein_list
    df_best["internal_stop"] = internal_stop_list

    # Keep only OTUs with high enough coverage, frame ambiguity and no internal stops
    df_best = df_best[
        (df_best.frame_ambiguity_frac >= cfg.frame_ambiguity_fraction)
        & (df_best.coverage >= cfg.min_query_coverage)
        & (~df_best.internal_stop)
    ]

    with (
        open(cfg.output_protein_fasta, "w") as f_prot,
        open(cfg.output_trimmed_fasta, "w") as f_trim,
    ):
        for seq_id, row in df_best.iterrows():
            f_prot.write(f">{seq_id}\n{row.protein}\n")
            f_trim.write(f">{seq_id}\n{row.trimmed_seq}\n")


if __name__ == "__main__":
    main()
