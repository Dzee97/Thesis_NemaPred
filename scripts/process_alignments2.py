from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
from common import parse_config
from scipy.stats import entropy as calc_entropy
from scipy.stats import zscore
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


def plot_infogain(infogain: np.ndarray, outfile: str):
    x = np.arange(len(infogain))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, infogain, alpha=0.8, s=6.0)

    ax.set_title("Column Infogain (Alignment Order)")
    ax.set_xlabel("Alignment Columns")
    ax.set_ylabel("Infogain (bits)")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def plot_infogain_std(infogain: np.ndarray, std: np.ndarray, outfile: str):
    x = infogain.flatten()
    y = std.flatten()

    mask = x > y

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x[mask], y[mask], alpha=0.3, s=6.0)
    ax.scatter(x[~mask], y[~mask], alpha=0.3, s=6.0)

    ax.set_title("Column Infogain VS Genus Entropy Standard Deviation")
    ax.set_xlabel("Infogain (bits)")
    ax.set_ylabel("Genus Entropy Std")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def plot_infogain_boxplots(infogain: np.ndarray, gene_start_idx: np.ndarray, outfile: str):
    if infogain.ndim == 1:
        infogain = infogain[None,]

    infogain_18s, infogain_28s, infogain_coi = np.split(infogain, gene_start_idx, axis=1)
    infogain_coi1, infogain_coi2, infogain_coi3 = (
        infogain_coi[:, 0::3],
        infogain_coi[:, 1::3],
        infogain_coi[:, 2::3],
    )

    data = [infogain_18s, infogain_28s, infogain_coi1, infogain_coi2, infogain_coi3]
    data = [array[~np.isnan(array)].flatten() for array in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    bplot = ax.boxplot(
        data,
        tick_labels=["18S", "28S", "COI 1st", "COI 2nd", "COI 3rd"],
        patch_artist=True,  # Enables filling boxes with color
        boxprops={"alpha": 0.7},
        medianprops={"color": "black", "linewidth": 1},
        flierprops={"marker": "o", "alpha": 0.3, "markersize": 4},  # Outliers
    )

    for patch, color in zip(
        bplot["boxes"], ["coral", "skyblue", "lightgreen", "lightgreen", "lightgreen"]
    ):
        patch.set_facecolor(color)

    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


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
) -> tuple[np.ndarray, np.ndarray]:
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

    return msa_joined, np.array([start_col_28s, start_col_coi])


def load_seq_dict(
    fasta_path: Path, constructor: type[skbio.DNA | skbio.Protein]
) -> dict[str, skbio.DNA | skbio.Protein]:
    return {
        seq.metadata["id"]: seq
        for seq in skbio.read(fasta_path, format="fasta", constructor=constructor)
    }


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
            "total_length",
        ],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    pivot_df["name"] = pivot_df.apply(
        lambda row: " ".join(
            [f"{gene}:{row.astype(str).accession[gene]}" for gene in ["18S", "28S", "COI"]]
        ),
        axis=1,
    )

    # Construct joined msa
    msa_joined, gene_start_idx = join_msas(
        pivot_df.accession,  # type: ignore
        msa_18s,
        msa_18s_index,
        msa_28s,
        msa_28s_index,
        msa_coi,
        msa_coi_index,
    )

    # Set non-degenerate characters and gaps to 0
    bases = ("A", "C", "G", "T")
    nondegen_mask = np.isin(msa_joined, [ord(b) for b in bases])
    msa_joined[~nondegen_mask] = 0

    # Find genus boundry indices
    genus_col = pivot_df.worms_genus.to_numpy()
    genus_start_idx = np.flatnonzero(genus_col[:-1] != genus_col[1:]) + 1

    # Greedy prefer longest p-distance clustering (99%)
    keep = np.ones(len(pivot_df), dtype=bool)
    for msa_genus, keep_genus in zip(
        np.split(msa_joined, genus_start_idx, axis=0),
        np.split(keep, genus_start_idx, axis=0),
    ):
        for i in range(msa_genus.shape[0] - 1):
            if not keep_genus[i]:
                continue

            shared_nondegen = (msa_genus[i] != 0) & (msa_genus[i + 1 :] != 0)
            mismatches = (msa_genus[i] != msa_genus[i + 1 :]) & shared_nondegen
            with np.errstate(invalid="ignore"):
                p_dist = np.sum(mismatches, axis=1) / np.sum(shared_nondegen, axis=1)

            keep_genus[i + 1 :] &= np.isnan(p_dist) | (p_dist > 0.01)

    # Filter out redundant sequences
    pivot_df = pivot_df[keep].reset_index(drop=True)
    msa_joined = msa_joined[keep]

    # Find genus boundry indices
    genus_col = pivot_df.worms_genus.to_numpy()
    genus_start_idx = np.flatnonzero(genus_col[:-1] != genus_col[1:]) + 1

    # Initialize arrays for genus nucleotide probabilities and entropies
    genus_probs = np.full((genus_start_idx.shape[0] + 1, len(bases), msa_joined.shape[1]), np.nan)
    genus_entropy = np.full((genus_start_idx.shape[0] + 1, msa_joined.shape[1]), np.nan)

    # Pseudocount parameter for Laplace smoothing
    alpha = 0.05

    # Minimum number of sequences covering a column in a genus
    min_seqs = 5

    # Loop over genera, calculate their nucleotide probabilities and entropies
    for msa_genus, probs, entropy in tqdm(
        zip(
            np.split(msa_joined, genus_start_idx, axis=0),
            genus_probs,
            genus_entropy,
        )
    ):
        base_counts = np.vstack([np.sum(msa_genus == ord(b), axis=0) for b in bases]) + alpha
        base_total = np.sum(base_counts, axis=0)
        base_total_mask = base_total >= min_seqs + alpha * len(bases)

        probs[:, base_total_mask] = base_counts[:, base_total_mask] / base_total[base_total_mask]

        entropy[base_total_mask] = calc_entropy(probs[:, base_total_mask], base=2, axis=0)

    # Initialize arrays for global entropies, genus infogain and genus entropy standard deviation
    global_entropy = np.full(msa_joined.shape[1], np.nan)
    genus_infogain = np.full(msa_joined.shape[1], np.nan)
    genus_entropy_std = np.full(msa_joined.shape[1], np.nan)

    # Minimum number of genera covering a column for infogain calculation
    min_genera = 5

    # Calculate the global entropy, genus infogain and genus entropy standard deviation
    num_genera = np.sum(~np.isnan(genus_entropy), axis=0)
    num_genera_mask = num_genera >= min_genera

    global_probs = np.nanmean(genus_probs[:, :, num_genera_mask], axis=0)
    global_entropy[num_genera_mask] = calc_entropy(global_probs, base=2, axis=0)  # type: ignore

    genus_infogain[num_genera_mask] = np.clip(
        global_entropy[num_genera_mask] - np.nanmean(genus_entropy[:, num_genera_mask], axis=0),
        a_min=0.0,
        a_max=2.0,
    )

    genus_entropy_std[num_genera_mask] = np.nanstd(genus_entropy[:, num_genera_mask], axis=0)

    # Plot class entropy grouped by gene
    plot_infogain_boxplots(global_entropy, gene_start_idx, "global_entropy_boxplot.png")

    # Plot class to genus infogain against genus entropy standard deviation
    plot_infogain_std(genus_infogain, genus_entropy_std, "genus_infogain_std_comp.png")

    # Discard columns where the genus entropy standard deviation is higher than the infogain
    infogain_mask = genus_infogain > genus_entropy_std
    genus_infogain[~infogain_mask] = np.nan

    # Plot the class to genus infogain
    plot_infogain_boxplots(genus_infogain, gene_start_idx, "genus_infogain_boxplot.png")

    # Calculate gain ratio by dividing infogain by class entropy
    genus_gainratio = genus_infogain.copy()
    genus_gainratio[infogain_mask] /= global_entropy[infogain_mask]

    # Plot the class to genus gainratio
    plot_infogain_boxplots(genus_gainratio, gene_start_idx, "genus_gainratio_boxplot.png")

    family_col = pivot_df.worms_family.to_numpy()
    family_start_idx = np.flatnonzero(family_col[:-1] != family_col[1:]) + 1

    all_summed_gainratio = np.full(len(pivot_df), np.nan)
    family_overlap_gainratio = np.full(len(pivot_df), np.nan)
    for msa_family, overlap_gainratio, summed_gainratio in zip(
        np.split(msa_joined, family_start_idx, axis=0),
        np.split(family_overlap_gainratio, family_start_idx, axis=0),
        np.split(all_summed_gainratio, family_start_idx, axis=0),
    ):
        nondegen = msa_family != 0
        msa_gainratio = nondegen * np.nan_to_num(genus_gainratio)

        pairwise_overlap_gainratio = nondegen @ msa_gainratio.T

        summed_gainratio[:] = np.diagonal(pairwise_overlap_gainratio)
        overlap_gainratio[:] = np.mean(pairwise_overlap_gainratio, axis=0)

    pivot_df["summed_gainratio"] = all_summed_gainratio
    pivot_df["family_overlap_gainratio"] = family_overlap_gainratio

    family_max = pivot_df.groupby("worms_genus").family_overlap_gainratio.transform("max")
    genus_rep = (pivot_df.family_overlap_gainratio > 0.9 * family_max) & (
        pivot_df.summed_gainratio > 100
    )

    pivot_df = pivot_df[genus_rep]

    pivot_df.to_csv(cfg.output_metadata)

    breakpoint()


if __name__ == "__main__":
    main()
