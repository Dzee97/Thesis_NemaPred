from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skbio
from skbio.alignment import align_dists
from skbio.tree import nj
from sklearn.metrics import silhouette_samples
from src.common import parse_config

TAXONOMY_COLS = ["worms_class", "worms_order", "worms_family", "worms_genus"]
SORTING_COLS = [
    *TAXONOMY_COLS,
    "worms_name",
    "clust_id",
    "length",
    "otu_coverage",
    "ambiguity_frac",
]
STATS_COLS = [
    *SORTING_COLS,
    "family_silhouette",
    "genus_median",
    "support_count",
    "support_members",
    "tip_name",
]


@dataclass
class Config:
    # Inputs / outputs
    input_parquet: Path
    input_trait_data: Path
    output_fasta: Path
    output_outgroup: Path
    output_genus_cov: Path
    output_ref_stats: Path

    # Technical sequence filters
    min_length: int
    max_ambiguity: float
    otu_coverage: float
    genus_length_frac: float

    # Molecular distance settings
    dist_model: str
    model_gamma: float
    max_z_score: float

    # Redundancy pruning filter
    prune_min_dist: float

    # Taxonomic outlier filter
    family_silhouette_margin: float


def create_compressed_alignment(df: pd.DataFrame) -> list[skbio.RNA]:
    """Reconstruct an alignment containing only columns used by these sequences."""
    positions_array = pa.array(df.msa_pos)
    combined_positions = positions_array.flatten()
    active_positions = combined_positions.unique().sort().to_numpy()

    aligned_sequences: list[skbio.RNA] = []
    for _, row in df.iterrows():
        aligned_bytes = np.full(len(active_positions), ord("-"), dtype=np.uint8)
        insert_indices = np.searchsorted(active_positions, row.msa_pos)
        aligned_bytes[insert_indices] = row.rna_seq
        aligned_sequences.append(skbio.RNA(aligned_bytes))

    return aligned_sequences


def add_coverage_column(genus_cov: pd.DataFrame, name: str, df_ref: pd.DataFrame) -> None:
    """Add one filtering-stage count column to the genus coverage table."""
    genus_cov[name] = df_ref.groupby(TAXONOMY_COLS).size().astype("Int64")


def mark_redundant(
    df_ref: pd.DataFrame, dist_data: np.ndarray, prune_min_dist: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mark genus-level redundant sequences for removal.

    A sequence can be absorbed when its nearest retained neighbour belongs
    to the same genus and either:

      - both records have the same WoRMS name, in which case the
        higher-quality sequence is retained; or

      - one record is unresolved at genus level and the other is resolved
        to species level, in which case the species-level record is retained.

    Different named species are never collapsed into each other.

    Sequence quality is compared lexicographically using:
      1. sequence length,
      2. OTU coverage,
      3. ambiguity fraction (lower is better).

    Returns
    -------
    keep
        Boolean mask of sequences retained by this pass.

    absorbed_by
        Positional index of the final surviving representative for every
        removed sequence, or -1 for retained sequences.
    """
    worms_genus = df_ref.worms_genus.to_numpy()
    worms_name = df_ref.worms_name.to_numpy()
    worms_rank = df_ref.worms_rank.to_numpy()
    length = df_ref.length.to_numpy()
    otu_coverage = df_ref.otu_coverage.to_numpy()
    ambiguity_frac = df_ref.ambiguity_frac.to_numpy()

    # Ignore self-distance when looking for the nearest retained neighbour.
    neighbour_dist = dist_data.copy()
    np.fill_diagonal(neighbour_dist, np.nan)

    keep = np.ones(len(df_ref), dtype=bool)
    absorbed_by = np.full(len(df_ref), -1, dtype=np.int32)

    while True:
        keep_idx = np.flatnonzero(keep)

        # Nothing can be redundant if only one sequence remains (will not happen in practice)
        if len(keep_idx) < 2:
            break

        for idx in keep_idx:
            local_closest = np.nanargmin(neighbour_dist[idx, keep_idx])
            closest_idx: np.intp = keep_idx[local_closest]

            if worms_genus[idx] != worms_genus[closest_idx]:
                continue

            if neighbour_dist[idx, closest_idx] > prune_min_dist:
                continue

            if worms_name[idx] == worms_name[closest_idx]:
                # Same taxon: keep the sequence that is more representative
                # of its genus in the current distance geometry.
                best_idx = max(
                    idx,
                    closest_idx,
                    key=lambda x: (length[x], otu_coverage[x], -ambiguity_frac[x]),
                )
            elif {worms_rank[idx], worms_rank[closest_idx]} == {"Genus", "Species"}:
                # Same genus, but one record is unresolved at genus level and
                # the other is resolved to species level: keep the species-level
                # record.
                best_idx = idx if worms_rank[idx] == "Species" else closest_idx
            else:
                # Two different named species are both informative.
                continue

            worst_idx = closest_idx if best_idx == idx else idx
            keep[worst_idx] = False
            absorbed_by[worst_idx] = best_idx
            break
        else:
            break

    # A sequence absorbed earlier in this pass may itself have been absorbed
    # later. Flatten A -> B -> C chains so every removed sequence points
    # directly to a final survivor.
    removed_idx = np.flatnonzero(~keep)
    for idx in removed_idx:
        absorb_idx = absorbed_by[idx]

        while absorbed_by[absorb_idx] != -1:
            absorb_idx = absorbed_by[absorb_idx]

        absorbed_by[idx] = absorb_idx

    # These assertions make the contract of this function explicit.
    assert np.all(absorbed_by[removed_idx] >= 0)
    assert np.all(keep[absorbed_by[removed_idx]])

    print(f"Mark {len(removed_idx)}, keep {keep.sum()}")
    return keep, absorbed_by


def remove_redundant(
    df_ref: pd.DataFrame,
    dist: skbio.DistanceMatrix,
    prune_min_dist: float,
    on_tree: bool = False,
) -> tuple[pd.DataFrame, skbio.DistanceMatrix]:
    """Repeatedly remove redundant sequences until this distance view is stable.

    When ``on_tree`` is False, redundancy is evaluated directly on the TN93
    distance matrix. When True, an NJ tree is rebuilt after every removal round
    and redundancy is evaluated on its cophenetic distances.

    Support from a redundant sequence is transferred to its surviving
    representative because the underlying molecular evidence is still valid.
    """
    while True:
        if on_tree:
            tree = nj(dist)
            tree_dist = tree.cophenet().filter(list(df_ref.index))
            dist_data = tree_dist.data
        else:
            dist_data = dist.data

        keep, absorbed_by = mark_redundant(df_ref, dist_data, prune_min_dist)

        if keep.all():
            return df_ref, dist

        # Every removed sequence now points directly to a final survivor, so
        # support can be transferred without depending on removal order.
        for remove_idx in np.flatnonzero(~keep):
            absorb_idx = absorbed_by[remove_idx]

            removed_accession = df_ref.index[remove_idx]  # type: ignore
            absorb_accession = df_ref.index[absorb_idx]

            members = (
                removed_accession,
                *df_ref.at[removed_accession, "support_members"],  # type: ignore
            )

            df_ref.at[absorb_accession, "support_members"] = (  # type: ignore
                df_ref.at[absorb_accession, "support_members"] + members  # type: ignore
            )

        df_ref = df_ref.iloc[np.flatnonzero(keep)].copy()
        dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index)))


def remove_redundancy_until_stable(
    df_ref: pd.DataFrame,
    dist: skbio.DistanceMatrix,
    genus_cov: pd.DataFrame,
    label: str,
    prune_min_dist: float,
) -> tuple[pd.DataFrame, skbio.DistanceMatrix]:
    """Alternate molecular- and NJ-based redundancy pruning until stable."""
    cycle = 0

    while True:
        cycle += 1
        n_before = len(df_ref)

        print(f"Cycle {cycle} - Remove redundant on molecular distance")
        df_ref, dist = remove_redundant(df_ref, dist, prune_min_dist, on_tree=False)
        add_coverage_column(
            genus_cov,
            f"{label} I{cycle} - molecular genus redundancy",
            df_ref,
        )

        print(f"Cycle {cycle} - Remove redundant on NJ cophenet")
        df_ref, dist = remove_redundant(df_ref, dist, prune_min_dist, on_tree=True)
        add_coverage_column(
            genus_cov,
            f"{label} I{cycle} - NJ genus redundancy",
            df_ref,
        )

        if len(df_ref) == n_before:
            return df_ref, dist


def remove_family_outliers(
    df_ref: pd.DataFrame,
    dist: skbio.DistanceMatrix,
    silhouette_margin: float,
) -> tuple[pd.DataFrame, skbio.DistanceMatrix]:
    """Remove strong family-placement outliers within genera.

    A sequence is removed when:
      1. its family silhouette is negative, meaning that it is on average
         closer to another family than to its own; and
      2. its family silhouette is at least ``silhouette_margin`` below the
         median family silhouette of sequences from the same genus.

    The comparison is therefore within genus. A genus whose sequences are
    collectively unusual at family level is retained, while isolated sequences
    that disagree strongly with the other representatives of their genus can
    be removed.

    The function never automatically removes every remaining sequence from
    a genus. Support from rejected sequences is not transferred because these
    sequences are treated as unreliable rather than redundant.
    """
    worms_family = df_ref.worms_family.to_numpy()
    worms_genus = df_ref.worms_genus.to_numpy()

    family_silhouette = np.asarray(
        silhouette_samples(
            dist.data,
            worms_family,
            metric="precomputed",
        )
    )

    # Typical family-placement quality for each genus.
    genus_median = np.full(len(df_ref), np.nan)

    for genus in np.unique(worms_genus):
        genus_idx = np.flatnonzero(worms_genus == genus)

        genus_median[genus_idx] = np.median(family_silhouette[genus_idx])

    # Strong sequence-level disagreement with the family placement
    # normally observed for this genus.
    remove = family_silhouette < genus_median - silhouette_margin

    flagged_count = remove.sum()

    # Never let an automatic taxonomy-based QC rule eliminate
    # every remaining representative of a genus.
    rescued_genera = 0

    for genus in np.unique(worms_genus):
        genus_idx = np.flatnonzero(worms_genus == genus)

        if np.all(remove[genus_idx]):
            best_idx = genus_idx[np.argmax(family_silhouette[genus_idx])]

            remove[best_idx] = False
            rescued_genera += 1

    print(
        f"Family outliers flagged: {flagged_count}, "
        f"removed: {remove.sum()}, "
        f"kept: {(~remove).sum()}, "
        f"genera safeguarded: {rescued_genera}"
    )

    df_ref["family_silhouette"] = family_silhouette
    df_ref["genus_median"] = genus_median

    df_ref = df_ref.loc[~remove]
    dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index)))

    return df_ref, dist


def main() -> None:
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    pa_table = pq.read_table(cfg.input_parquet)
    df: pd.DataFrame = pa_table.to_pandas(
        types_mapper=pd.ArrowDtype,
        ignore_metadata=True,
    )
    df.set_index("accession", inplace=True)

    df_ref = df[df.seq_type == "ref"].copy()

    df_trait_genera = pd.read_csv(cfg.input_trait_data, usecols=TAXONOMY_COLS).drop_duplicates()

    # ------------------------------------------------------------------
    # 1. Context taxa: marine genera from the same orders as trait genera
    # ------------------------------------------------------------------
    df_ref = df_ref[
        df_ref.worms_genus.notna()
        & df_ref.worms_ismarine.fillna(False)
        & df_ref.worms_order.isin(df_trait_genera.worms_order)
    ].copy()

    genus_cov_index = pd.MultiIndex.from_frame(df_trait_genera).union(
        pd.MultiIndex.from_frame(df_ref[TAXONOMY_COLS].drop_duplicates())
    )
    genus_cov = pd.DataFrame(index=genus_cov_index)
    genus_cov["traits_avail"] = genus_cov.index.get_level_values("worms_genus").isin(
        df_trait_genera.worms_genus
    )
    genus_cov.sort_values(
        by=["traits_avail"] + TAXONOMY_COLS,
        ascending=[False] + [True] * len(TAXONOMY_COLS),
        inplace=True,
    )
    add_coverage_column(genus_cov, "NCBI + SILVA", df_ref)

    # ------------------------------------------------------------------
    # 2. Technical sequence QC
    # ------------------------------------------------------------------
    df_ref = df_ref[df_ref.length >= cfg.min_length].copy()
    add_coverage_column(genus_cov, f"Length >= {cfg.min_length}", df_ref)

    df_ref = df_ref[df_ref.ambiguity_frac <= cfg.max_ambiguity].copy()
    add_coverage_column(genus_cov, f"Ambiguity <= {cfg.max_ambiguity}", df_ref)

    df_ref = df_ref[df_ref.otu_coverage >= cfg.otu_coverage].copy()
    add_coverage_column(genus_cov, f"OTU coverage >= {cfg.otu_coverage}", df_ref)

    # ------------------------------------------------------------------
    # 3. One representative per WoRMS name x global 99% cluster
    # ------------------------------------------------------------------
    df_ref = df_ref.sort_values(
        SORTING_COLS,
        ascending=[True, True, True, True, True, True, False, False, True],
    )
    df_ref = df_ref.groupby(["worms_name", "clust_id"], sort=False, group_keys=False).head(1).copy()
    add_coverage_column(genus_cov, "WoRMS name x global cluster", df_ref)

    # ------------------------------------------------------------------
    # 3. Remove genus-level sequences duplicated by species-level
    # ------------------------------------------------------------------
    df_ref["seq_bytes"] = df_ref.rna_seq.apply(lambda x: x.tobytes())
    remove = []
    for _, duplicate_group in df_ref.groupby("seq_bytes", sort=False):
        if len(duplicate_group) < 2:
            continue

        has_species = (duplicate_group.worms_rank == "Species").any()
        if not has_species:
            continue

        remove.extend(duplicate_group[duplicate_group.worms_rank == "Genus"].index.to_list())

    df_ref = df_ref.drop(columns="seq_bytes")
    df_ref = df_ref.drop(labels=remove)
    add_coverage_column(genus_cov, "Genus-Species deduplication", df_ref)

    # ------------------------------------------------------------------
    # 3. Genus-relative length filter
    # ------------------------------------------------------------------
    df_ref = df_ref[
        df_ref.length
        >= cfg.genus_length_frac * df_ref.groupby("worms_genus").length.transform("max")
    ].copy()
    add_coverage_column(genus_cov, f"Length >= {cfg.genus_length_frac} x genus max", df_ref)

    # ------------------------------------------------------------------
    # 4. Compute molecular distances and remove extreme global outliers
    # ------------------------------------------------------------------
    msa = skbio.TabularMSA(create_compressed_alignment(df_ref), index=df_ref.index)
    full_dist = align_dists(
        msa,
        metric=cfg.dist_model,
        gamma=cfg.model_gamma,
        shared_by_all=False,
    )

    dist = full_dist
    dist.data[dist.data < 0.0] = 0.0

    dist_data = dist.data.copy()
    np.fill_diagonal(dist_data, np.nan)

    median_dist = np.nanmedian(dist_data, axis=1)
    median_std = median_dist.std()
    if median_std == 0:
        z_scores = np.zeros(len(median_dist))
    else:
        z_scores = (median_dist - median_dist.mean()) / median_std

    keep_idx = np.flatnonzero(z_scores <= cfg.max_z_score)
    df_ref = df_ref.iloc[keep_idx].copy()
    dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index)))
    add_coverage_column(genus_cov, f"Distance Z-score <= {cfg.max_z_score}", df_ref)

    # Each sequence now represents one distinct post-QC WoRMS-name x cluster
    # observation. Redundancy pruning below will transfer these members rather
    # than simply counting raw database accessions.
    df_ref["support_members"] = [() for _ in range(len(df_ref))]

    # ------------------------------------------------------------------
    # 5. Compress redundant information before taxonomy-based QC
    # ------------------------------------------------------------------
    df_ref, dist = remove_redundancy_until_stable(
        df_ref, dist, genus_cov, label="Pre-family", prune_min_dist=cfg.prune_min_dist
    )

    # ------------------------------------------------------------------
    # 6. One conservative family/genus outlier pass
    # ------------------------------------------------------------------
    df_ref, dist = remove_family_outliers(
        df_ref,
        dist,
        silhouette_margin=cfg.family_silhouette_margin,
    )
    add_coverage_column(genus_cov, "Family/genus outliers", df_ref)

    # Removing outliers can make previously non-neighbouring sequences become
    # redundant, so allow redundancy pruning to converge again. We deliberately
    # do not iterate the taxonomy-based outlier filter itself.
    df_ref, dist = remove_redundancy_until_stable(
        df_ref, dist, genus_cov, label="Post-family", prune_min_dist=cfg.prune_min_dist
    )

    # ------------------------------------------------------------------
    # 7. Keep only genera with traits and compress redundancy once more
    # ------------------------------------------------------------------
    df_ref = df_ref[df_ref.worms_genus.isin(df_trait_genera.worms_genus)].copy()
    dist = cast(skbio.DistanceMatrix, dist.filter(list(df_ref.index)))
    add_coverage_column(genus_cov, "Trait genera only", df_ref)

    df_ref, dist = remove_redundancy_until_stable(
        df_ref, dist, genus_cov, label="Trait only", prune_min_dist=cfg.prune_min_dist
    )

    # The surviving sequence itself is one unit of support in addition to all
    # representatives that were absorbed into it.
    df_ref["support_count"] = df_ref.support_members.apply(len) + 1

    # ------------------------------------------------------------------
    # 9. Give unique tip name for every surviving sequence
    # ------------------------------------------------------------------
    df_ref["tip_name"] = df_ref["worms_name"].copy()
    df_ref.loc[df_ref.worms_rank == "Genus", "tip_name"] += " sp."

    multiple_seqs = df_ref.groupby("worms_name").worms_name.transform("size") > 1
    unassigned_nums = df_ref[multiple_seqs].groupby("worms_name").cumcount() + 1

    df_ref.loc[multiple_seqs, "tip_name"] += " " + unassigned_nums.astype(str)

    # ------------------------------------------------------------------
    # 8. Save diagnostics and candidate alignment
    # ------------------------------------------------------------------
    ref_stats = df_ref[STATS_COLS].copy()

    # CSV-friendly representation of the supporting accessions.
    ref_stats["support_members"] = ref_stats.support_members.apply(
        lambda members: ";".join(map(str, members))
    )

    ref_stats.to_csv(cfg.output_ref_stats)
    genus_cov.to_csv(cfg.output_genus_cov)

    df_outgroup = df[df.seq_type == "outgroup"].copy()
    alignment_accessions = df_outgroup.index.to_list() + df_ref.index.to_list()
    outgroup_names = df_outgroup.worms_name.to_list()
    tip_names = outgroup_names + df_ref.tip_name.to_list()

    df_alignment = df[["rna_seq", "msa_pos"]].loc[alignment_accessions]
    aligned_sequences = create_compressed_alignment(df_alignment)

    newick_compat = str.maketrans({" ": "_", "(": "", ")": ""})
    with open(cfg.output_fasta, "w") as f:
        for tip_name, seq in zip(tip_names, aligned_sequences):
            tip_name = tip_name.translate(newick_compat)
            f.write(f">{tip_name}\n{seq}\n")

    with open(cfg.output_outgroup, "w") as f:
        f.write(",".join(name.translate(newick_compat) for name in outgroup_names))


if __name__ == "__main__":
    main()
