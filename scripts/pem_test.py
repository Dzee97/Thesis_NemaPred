import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import skbio
from common import parse_config
from joblib import parallel_backend
from optuna.distributions import FloatDistribution
from optuna.exceptions import ExperimentalWarning
from optuna_integration import OptunaSearchCV
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.callback import ProgressBar
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


@dataclass
class Config:
    input_tree: Path
    input_trait_data: Path
    trait_column: str
    k_folds: int
    repeats: int
    sweep_points: int
    steepness: float
    regularization_strength: float
    l1_ratio: float
    train_epochs: int
    learning_rate: float
    seed: int
    output_sweep: Path
    output_predictions: Path
    output_summary: Path
    output_plot: Path


def create_influence_matrix(
    tree: skbio.TreeNode, taxa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    taxon_index = {name: i for i, name in enumerate(taxa)}

    edges = [node for node in tree.postorder() if node.parent is not None]
    branch_lengths = np.empty(len(edges), dtype=float)
    influence = np.zeros((len(taxa), len(edges)), dtype=float)

    for j, node in enumerate(edges):
        if node.length is None:
            raise ValueError("All tree edges must have branch lengths for PEM")
        if node.length < 0:
            raise ValueError("PEM cannot use negative branch lengths")

        branch_lengths[j] = node.length

        for tip in node.tips(include_self=True):
            i = taxon_index.get(str(tip.name))
            if i is not None:
                influence[i, j] = 1.0

    return influence, branch_lengths


class PEMTransformer(BaseEstimator, TransformerMixin):
    """Build a PEM on training taxa and project held-out taxa onto it."""

    def __init__(self, branch_lengths: np.ndarray, steepness: float = 0.0):
        self.branch_lengths = branch_lengths
        self.steepness = steepness

    def fit(self, X: np.ndarray, y=None):
        # Remove branches that only lead to held-out taxa.
        self.active_edges_ = np.any(X > 0, axis=0)
        B_train = X[:, self.active_edges_]
        phi = self.branch_lengths[self.active_edges_]

        # Standard PEM edge weighting with psi = 1.
        self.edge_weights_ = np.zeros_like(phi)
        positive = phi > 0
        self.edge_weights_[positive] = phi[positive] ** ((1.0 - self.steepness) / 2.0)

        weighted_train = B_train * self.edge_weights_
        self.column_mean_ = weighted_train.mean(axis=0)
        X_centered = weighted_train - self.column_mean_

        U, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
        if singular_values.size == 0:
            raise ValueError("No PEM axes could be calculated from this training tree")

        tolerance = np.finfo(float).eps * max(X_centered.shape) * singular_values
        keep = singular_values > tolerance
        if not np.any(keep):
            raise ValueError("Training tree has no non-zero PEM axes")

        # Store projection operator V S^-1 for unseen taxa
        self.projection_matrix_ = Vt[keep].T / singular_values[keep]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        B = X[:, self.active_edges_]
        weighted = B * self.edge_weights_
        X_centered = weighted - self.column_mean_
        return (X_centered @ self.projection_matrix_).astype(np.float32)


def main() -> None:
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    print("Loading tree and trait data...")
    tree: skbio.TreeNode = skbio.read(cfg.input_tree, format="newick", into=skbio.TreeNode)  # type: ignore
    trait_df = pd.read_csv(cfg.input_trait_data)

    if cfg.trait_column not in trait_df.columns:
        raise ValueError(f"Trait column not found: {cfg.trait_column}")

    trait_series = trait_df.set_index("worms_genus")[cfg.trait_column].dropna()
    taxa = np.array([str(tip.name) for tip in tree.tips() if str(tip.name) in trait_series.index])
    trait_values = trait_series.loc[taxa]

    le = LabelEncoder()
    y = np.asarray(le.fit_transform(trait_values))

    influence, branch_lengths = create_influence_matrix(tree, taxa)

    # Set up Repeated Stratified K-Fold CV
    cv = RepeatedStratifiedKFold(n_splits=cfg.k_folds, n_repeats=cfg.repeats, random_state=cfg.seed)

    param_distributions = {
        "pem__steepness": FloatDistribution(0.0, 1.0),
        "clf__C": FloatDistribution(1e-4, 1e4, log=True),
        "clf__l1_ratio": FloatDistribution(0.0, 1.0),
    }

    n_iter = max(cfg.sweep_points, 60)

    print(f"Running RandomizedSearchCV ({n_iter} iterations)...")
    pipe = Pipeline(
        [
            ("pem", PEMTransformer(branch_lengths=branch_lengths)),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=2000,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )

    pbar = tqdm(total=n_iter, desc="Optuna Trials")

    def tqdm_callback(study, trial):
        pbar.update(1)
        pbar.set_postfix(best_score=f"{study.best_value:.4f}")

    # Threading backend avoids pickling __main__.PEMTransformer under Snakemake
    with parallel_backend("threading"), warnings.catch_warnings():
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        search = OptunaSearchCV(
            estimator=pipe,
            param_distributions=param_distributions,
            n_trials=n_iter,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=cfg.seed,
            refit=True,
            return_train_score=True,
            callbacks=[tqdm_callback],
            verbose=0,
        )

        search.fit(influence, y)

    print(f"Best CV Loss: {search.best_score_:.3f}")
    print(f"Best Hyperparameters: {search.best_params_}")

    # 1. Save full hyperparameter sweep results
    Path(cfg.output_sweep).parent.mkdir(parents=True, exist_ok=True)
    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_df.to_csv(cfg.output_sweep, index=False)

    # 2. Collect out-of-fold predictions across all repeats using the best hyperparameters
    best_params = search.best_params_
    best_pipe = Pipeline(
        [
            (
                "pem",
                PEMTransformer(
                    branch_lengths=branch_lengths,
                    steepness=best_params["pem__steepness"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    C=best_params["clf__C"],
                    l1_ratio=best_params["clf__l1_ratio"],
                    max_iter=2000,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )

    oof_records = []
    for split_idx, (train_idx, test_idx) in enumerate(cv.split(influence, y)):
        repeat_idx = split_idx // cfg.k_folds
        fold_idx = split_idx % cfg.k_folds

        best_pipe.fit(influence[train_idx], y[train_idx])
        preds = best_pipe.predict(influence[test_idx])
        probs = best_pipe.predict_proba(influence[test_idx])

        for i, idx in enumerate(test_idx):
            record = {
                "taxon": taxa[idx],
                "repeat": repeat_idx,
                "fold": fold_idx,
                "true_label": le.inverse_transform([y[idx]]),
                "predicted_label": le.inverse_transform([preds[i]]),
                "true_encoded": int(y[idx]),
                "pred_encoded": int(preds[i]),
            }
            for class_idx, class_name in enumerate(le.classes_):
                record[f"prob_{class_name}"] = float(probs[i, class_idx])
            oof_records.append(record)

    Path(cfg.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(oof_records)
    predictions_df.to_csv(cfg.output_predictions, index=False)

    # 3. Save summary metrics
    overall_acc = accuracy_score(predictions_df["true_encoded"], predictions_df["pred_encoded"])
    overall_f1 = f1_score(
        predictions_df["true_encoded"],
        predictions_df["pred_encoded"],
        average="weighted",
    )

    Path(cfg.output_summary).parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(
        [
            {
                "trait": cfg.trait_column,
                "n_taxa": len(taxa),
                "k_folds": cfg.k_folds,
                "repeats": cfg.repeats,
                "n_iter": n_iter,
                "best_steepness": best_params["pem__steepness"],
                "best_C": best_params["clf__C"],
                "best_l1_ratio": best_params["clf__l1_ratio"],
                "best_cv_accuracy": search.best_score_,
                "oof_accuracy": overall_acc,
                "oof_weighted_f1": overall_f1,
            }
        ]
    )
    summary_df.to_csv(cfg.output_summary, index=False)

    # 4. Save confusion matrix plot
    Path(cfg.output_plot).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(predictions_df["true_encoded"], predictions_df["pred_encoded"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(
        f"{cfg.trait_column} (steepness={best_params['pem__steepness']:.2f}, "
        f"C={best_params['clf__C']:.2e}, l1={best_params['clf__l1_ratio']:.2f})"
    )
    plt.tight_layout()
    plt.savefig(cfg.output_plot, dpi=300)
    plt.close(fig)

    print("Done! All outputs saved.")


if __name__ == "__main__":
    main()
