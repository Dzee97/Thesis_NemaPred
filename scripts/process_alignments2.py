import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
from common import parse_config
from matplotlib.axes import Axes
from scipy.stats import entropy
from tqdm import tqdm


@dataclass
class Config:
    input_metadata: Path
    input_18s_ref_fasta: Path
    input_18s_otu_fasta: Path
    input_28s_ref_fasta: Path
    input_coi_ref_and_otu_prot_fasta: Path
    input_coi_ref_trimmed_fasta: Path
    input_coi_otu_trimmed_fasta: Path
    output_fasta: Path
    output_metadata: Path


def create_minimal_msa(
    seq_dict: dict[str, skbio.RNA | skbio.Protein],
    nuc_seqs: None | dict[str, skbio.DNA] = None,
) -> tuple[dict[str, int], np.ndarray]:

    alignment_length = -1
    unique_positions = set()

    for seq in seq_dict.values():
        if alignment_length < 0:
            alignment_length = len(seq)

        non_gap_mask = ~seq.gaps()
        positions = np.flatnonzero(non_gap_mask).astype(np.uint32)
        unique_positions |= set(positions)

    unique_positions = np.sort(list(unique_positions))
    new_position_map = np.full(alignment_length, -1, dtype=np.int32)
    new_position_map[unique_positions] = np.arange(len(unique_positions))

    msa_index = {}

    msa_length = len(unique_positions) if nuc_seqs is None else len(unique_positions) * 3
    msa = np.full((len(seq_dict), msa_length), ord("-"), dtype=np.uint8)

    for i, (accession, seq) in enumerate(seq_dict.items()):
        msa_index[accession] = i

        non_gap_mask = ~seq.gaps()
        new_positions = new_position_map[non_gap_mask]

        if nuc_seqs is not None:
            new_positions = new_positions * 3
            new_positions = np.sort(
                np.concatenate((new_positions, new_positions + 1, new_positions + 2))
            )

            nuc_seq = nuc_seqs[accession]
        else:
            nuc_seq = seq

        ungapped_seq = nuc_seq.degap().values.view(np.uint8)
        msa[i, new_positions] = ungapped_seq

    return msa_index, msa


def filter_msa(
    msa_index: dict[str, int], msa: np.ndarray, filter_ids: list[str]
) -> tuple[dict[str, int], np.ndarray]:
    filter_idx = [msa_index[id] for id in filter_ids]
    new_msa_index = {id: i for i, id in enumerate(filter_ids)}
    new_msa = msa[filter_idx]

    return new_msa_index, new_msa


def join_msas(
    matched_accessions: pd.DataFrame,
    msa_18s: np.ndarray,
    msa_18s_index: dict[str, int],
    msa_28s: np.ndarray,
    msa_28s_index: dict[str, int],
    msa_coi: np.ndarray,
    msa_coi_index: dict[str, int],
) -> np.ndarray:
    msa_18s_length, msa_28s_length, msa_coi_length = (
        msa_18s.shape[1],
        msa_28s.shape[1],
        msa_coi.shape[1],
    )
    msa_joined_length = msa_18s_length + msa_28s_length + msa_coi_length
    msa_joined = np.full((len(matched_accessions), msa_joined_length), ord("-"), dtype=np.uint8)

    start_col_18s, end_col_18s = 0, msa_18s_length
    start_col_28s, end_col_28s = msa_18s_length, msa_18s_length + msa_28s_length
    start_col_coi, end_col_coi = msa_18s_length + msa_28s_length, msa_joined_length

    for gene, start_col, end_col, msa_index, msa_matrix in (
        ("18S", start_col_18s, end_col_18s, msa_18s_index, msa_18s),
        ("28S", start_col_28s, end_col_28s, msa_28s_index, msa_28s),
        ("COI", start_col_coi, end_col_coi, msa_coi_index, msa_coi),
    ):
        insert_series = matched_accessions[gene].dropna()
        insert_idx = insert_series.index.to_list()
        insert_ids = insert_series.to_list()

        _, insert_matrix = filter_msa(msa_index, msa_matrix, insert_ids)
        msa_joined[insert_idx, start_col:end_col] = insert_matrix

    return msa_joined


def load_seq_dict(
    fasta_path: Path, constructor: type[skbio.DNA | skbio.Protein]
) -> dict[str, skbio.DNA | skbio.Protein]:
    return {
        seq.metadata["id"]: seq
        for seq in skbio.read(fasta_path, format="fasta", constructor=constructor)
    }


def calculate_smoothed_entropy(elements: np.ndarray, num_categories: int, alpha=0.5) -> float:
    denominator = len(elements) + (alpha * num_categories)
    _, counts = np.unique(elements, return_counts=True)
    full_counts = np.zeros(num_categories, dtype=int)
    full_counts[: len(counts)] = counts
    probs = (full_counts + alpha) / denominator
    return float(entropy(probs, base=2))


def main() -> None:
    # Load snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    # Load 18S alignment
    seqs_18s_ref = load_seq_dict(cfg.input_18s_ref_fasta, skbio.DNA)
    msa_18s_index, msa_18s = create_minimal_msa(seqs_18s_ref)

    # Load 28S alignment
    seqs_28s_ref = load_seq_dict(cfg.input_28s_ref_fasta, skbio.DNA)
    msa_28s_index, msa_28s = create_minimal_msa(seqs_28s_ref)

    # Load COI alignment
    prot_seqs_coi = load_seq_dict(cfg.input_coi_ref_and_otu_prot_fasta, skbio.Protein)
    nuc_seqs_coi_ref = load_seq_dict(cfg.input_coi_ref_trimmed_fasta, skbio.DNA)
    prot_seqs_coi_ref = {id: prot_seqs_coi[id] for id in nuc_seqs_coi_ref}
    msa_coi_index, msa_coi = create_minimal_msa(prot_seqs_coi_ref, nuc_seqs=nuc_seqs_coi_ref)

    # Load metadata table
    df = pd.read_csv(cfg.input_metadata)

    # Normalize voucher names
    voucher_trans = str.maketrans({" ": "", "-": "", "_": ""})
    df["voucher_norm"] = df.apply(
        lambda row: (
            row.fillna("")
            .voucher.lower()
            .removesuffix("18s")
            .removesuffix("28s")
            .removesuffix("coi")
            .translate(voucher_trans)
            .removesuffix(row.worms_name.lower().translate(voucher_trans))
        ),
        axis=1,
    )

    no_voucher = df.voucher_norm == ""
    df.loc[no_voucher, "voucher_norm"] = df.loc[no_voucher, "accession"]

    # Pivot the table by shared voucher
    pivot_df = df.pivot_table(
        index=[
            "worms_class",
            "worms_order",
            "worms_family",
            "worms_genus",
            "worms_name",
            "worms_rank",
            "trait_genus",
            "voucher_norm",
        ],
        columns="query_gene",
        values=["accession", "length"],
        aggfunc="first",
        sort=False,
    ).reset_index()

    pivot_df["total_length"] = pivot_df.length.fillna(0).sum(axis=1)  # type: ignore

    pivot_df = pivot_df.sort_values(
        [
            "worms_class",
            "worms_order",
            "worms_family",
            "worms_genus",
            "worms_name",
            "worms_rank",
            "trait_genus",
            "total_length",
            "voucher_norm",
        ],
        ascending=[True, True, True, True, True, True, True, False, True],
    ).reset_index(drop=True)

    pivot_df["name"] = pivot_df.apply(
        lambda row: " ".join(
            [f"{gene}:{row.astype(str).accession[gene]}" for gene in ["18S", "28S", "COI"]]
        ),
        axis=1,
    )

    # Construct joined msa
    msa_joined = join_msas(
        pivot_df.accession,  # type: ignore
        msa_18s,
        msa_18s_index,
        msa_28s,
        msa_28s_index,
        msa_coi,
        msa_coi_index,
    )

    nondegen_mask = np.isin(msa_joined, [ord(c) for c in ["A", "C", "G", "T"]])

    worms_family = pivot_df.worms_order.to_numpy()
    worms_genus = pivot_df.worms_genus.to_numpy()
    worms_name = pivot_df.worms_name.to_numpy()
    worms_rank = pivot_df.worms_rank.to_numpy()
    total_length = pivot_df.total_length.to_numpy()

    for family in np.unique(worms_family):
        print(family)

        family_idx = np.flatnonzero(worms_family == family)

        family_worms_genus = worms_genus[family_idx]
        family_worms_genus_unique = np.unique(family_worms_genus)
        if len(family_worms_genus_unique) < 2:
            continue

        family_worms_name = worms_name[family_idx]
        family_is_species = worms_rank[family_idx] == "Species"

        family_col_weights = np.zeros(msa_joined.shape[1])
        for col_idx in tqdm(range(msa_joined.shape[1])):
            family_col_nondegen_idx = np.flatnonzero(nondegen_mask[family_idx, col_idx])
            if len(family_col_nondegen_idx) < 10:
                continue

            family_col_worms_genus = family_worms_genus[family_col_nondegen_idx]
            family_col_worms_genus_unique = np.unique(family_col_worms_genus)
            if len(family_col_worms_genus_unique) < 2:
                continue

            family_col_vals = msa_joined[family_idx, col_idx][family_col_nondegen_idx]
            family_col_worms_name = family_worms_name[family_col_nondegen_idx]
            family_col_rep_mask = ~family_is_species[family_col_nondegen_idx]

            for species in np.unique(family_col_worms_name[~family_col_rep_mask]):
                species_idx = np.flatnonzero(family_col_worms_name == species)
                species_col_vals = family_col_vals[species_idx]

                vals, counts = np.unique(species_col_vals, return_counts=True)
                max_idx = np.argmax(counts)
                max_val = vals[max_idx]

                rep_species_idx = np.argmax(species_col_vals == max_val)
                family_col_rep_mask[species_idx[rep_species_idx]] = True

            family_col_rep_vals = family_col_vals[family_col_rep_mask]
            family_col_entropy = calculate_smoothed_entropy(family_col_rep_vals, num_categories=4)

            family_col_rep_genus = family_col_worms_genus[family_col_rep_mask]
            family_col_infogain = family_col_entropy
            for genus in family_col_worms_genus_unique:
                genus_idx = np.flatnonzero(family_col_rep_genus == genus)
                genus_col_rep_vals = family_col_rep_vals[genus_idx]
                genus_col_entropy = calculate_smoothed_entropy(genus_col_rep_vals, num_categories=4)

                family_col_infogain -= (
                    len(genus_col_rep_vals) / len(family_col_rep_vals) * genus_col_entropy
                )

            family_col_intrinsic = calculate_smoothed_entropy(
                family_col_rep_genus, num_categories=len(family_worms_genus_unique)
            )
            family_col_weights[col_idx] = np.clip(family_col_infogain, a_min=0.0, a_max=None) / (
                family_col_intrinsic * family_col_entropy
            )

        family_msa = msa_joined[family_idx]
        family_nondegen_mask = nondegen_mask[family_idx]

        family_mismatches = family_msa[:, None, :] != family_msa[None, :, :]
        family_shared_nondegen = family_nondegen_mask[:, None, :] & family_nondegen_mask[None, :, :]
        family_nondegen_mismatches = family_mismatches & family_shared_nondegen

        family_weighted_p_dist = np.sum(
            family_nondegen_mismatches * family_col_weights, axis=2
        ) / np.sum(family_shared_nondegen * family_col_weights, axis=2)
        np.fill_diagonal(family_weighted_p_dist, np.nan)

        for genus in family_worms_genus_unique:
            genus_idx = np.flatnonzero(family_worms_genus == genus)
            genus_weighted_p_dist = family_weighted_p_dist[np.ix_(genus_idx, genus_idx)]
            low_dist = genus_weighted_p_dist < 0.05
            low_dist[np.tril_indices_from(low_dist)] = False
            breakpoint()


if __name__ == "__main__":
    main()
