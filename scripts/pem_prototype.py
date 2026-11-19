from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
import torch
import torch.nn.functional as F
from common import parse_config
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold


@dataclass
class Config:
    input_tree: Path
    input_trait_data: Path
    trait_column: str
    k_folds: int
    repeats: int
    bo_iterations: int
    train_epochs: int
    learning_rate: float
    seed: int
    output_optimization: Path
    output_predictions: Path
    output_summary: Path
    output_plot: Path


def create_influence_matrix(tree: skbio.TreeNode, taxa: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Create the binary tip x edge PEM influence matrix."""
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

        # An influence-matrix entry is 1 when the edge is ancestral to the tip.
        for tip in node.tips(include_self=True):
            i = taxon_index.get(tip.name)
            if i is not None:
                influence[i, j] = 1.0

    return influence, branch_lengths


def calculate_pem_features(
    influence_train: np.ndarray,
    influence_test: np.ndarray,
    branch_lengths: np.ndarray,
    steepness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a PEM on the training genera and project held-out genera onto it."""
    # Remove branches that only lead to held-out taxa. This is equivalent to
    # pruning the held-out terminal lineages while retaining their placement node.
    active_edges = np.any(influence_train > 0, axis=0)
    B_train = influence_train[:, active_edges]
    B_test = influence_test[:, active_edges]
    phi = branch_lengths[active_edges]

    # Standard PEM edge weighting with psi = 1.
    edge_weights = np.zeros_like(phi)
    positive = phi > 0
    edge_weights[positive] = phi[positive] ** ((1.0 - steepness) / 2.0)

    weighted_train = B_train * edge_weights
    weighted_test = B_test * edge_weights

    # PEMs are obtained from the centred weighted influence matrix.
    column_mean = weighted_train.mean(axis=0)
    X_train = weighted_train - column_mean
    X_test = weighted_test - column_mean

    U, singular_values, Vt = np.linalg.svd(X_train, full_matrices=False)

    # Numerical rank. A centred n x m matrix has rank at most n - 1.
    if singular_values.size == 0:
        raise ValueError("No PEM axes could be calculated from this training tree")

    tolerance = np.finfo(float).eps * max(X_train.shape) * singular_values[0]
    keep = singular_values > tolerance

    if not np.any(keep):
        raise ValueError("Training tree has no non-zero PEM axes")

    # The training PEM scores are the left singular vectors U.
    train_features = U[:, keep] * singular_values[keep]

    # From X = U S V^T, a new centred influence row x is projected as
    # u_new = x V S^-1.
    test_features = X_test @ Vt[keep].T

    return train_features.astype(np.float32), test_features.astype(np.float32)


def train_elastic_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    regularization_strength: float,
    l1_ratio: float,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit multiclass logistic regression with an elastic-net penalty in PyTorch."""
    torch.manual_seed(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = torch.nn.Linear(X_train.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        logits = model(X_train_t)
        cross_entropy = F.cross_entropy(logits, y_train_t)

        # Do not regularize the intercept.
        weights = model.weight
        l1_penalty = torch.sum(torch.abs(weights))
        l2_penalty = 0.5 * torch.sum(weights**2)
        elastic_net = regularization_strength * (
            l1_ratio * l1_penalty + (1.0 - l1_ratio) * l2_penalty
        )

        loss = cross_entropy + elastic_net
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X_test_t)
        validation_loss = F.cross_entropy(logits, y_test_t).item()
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)

    return validation_loss, predictions, probabilities


def evaluate_parameters(
    steepness: float,
    regularization_strength: float,
    l1_ratio: float,
    influence: np.ndarray,
    branch_lengths: np.ndarray,
    y: np.ndarray,
    taxa: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    cfg: Config,
    return_predictions: bool = False,
) -> tuple[float, pd.DataFrame | None]:
    fold_losses = []
    prediction_rows = []

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = calculate_pem_features(
            influence[train_idx],
            influence[test_idx],
            branch_lengths,
            steepness,
        )

        validation_loss, predictions, probabilities = train_elastic_net(
            X_train=X_train,
            y_train=y[train_idx],
            X_test=X_test,
            y_test=y[test_idx],
            n_classes=n_classes,
            regularization_strength=regularization_strength,
            l1_ratio=l1_ratio,
            epochs=cfg.train_epochs,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed + split_idx,
        )
        fold_losses.append(validation_loss)

        if return_predictions:
            repeat = split_idx // cfg.k_folds
            fold = split_idx % cfg.k_folds

            for row_idx, taxon_idx in enumerate(test_idx):
                row = {
                    "repeat": repeat + 1,
                    "fold": fold + 1,
                    "genus": taxa[taxon_idx],
                    "true_class": int(y[taxon_idx]),
                    "predicted_class": int(predictions[row_idx]),
                }
                for class_idx, probability in enumerate(probabilities[row_idx]):
                    row[f"probability_{class_idx}"] = probability
                prediction_rows.append(row)

    predictions_df = pd.DataFrame(prediction_rows) if return_predictions else None
    return float(np.mean(fold_losses)), predictions_df


def decode_search_point(x: np.ndarray) -> tuple[float, float, float]:
    """Map a point in the unit cube to the three model hyperparameters."""
    steepness = float(x[0])
    log10_regularization = -6.0 + 6.0 * float(x[1])
    regularization_strength = 10.0**log10_regularization
    l1_ratio = float(x[2])

    return steepness, regularization_strength, l1_ratio


def expected_improvement(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    best_loss: float,
) -> np.ndarray:
    mean, std = gp.predict(X, return_std=True)
    std = np.maximum(std, 1e-12)

    improvement = best_loss - mean
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def bayesian_optimization(
    objective,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Small Gaussian-process Bayesian optimizer over a 3-D unit cube."""
    if iterations < 1:
        raise ValueError("bo_iterations must be at least 1")

    rng = np.random.default_rng(seed)
    n_initial = min(6, iterations)

    X_observed = np.full((1, 3), 0.5)
    if n_initial > 1:
        X_observed = np.vstack([X_observed, rng.uniform(0.0, 1.0, size=(n_initial - 1, 3))])
    y_observed = []

    for i, x in enumerate(X_observed):
        steepness, regularization_strength, l1_ratio = decode_search_point(x)
        loss = objective(steepness, regularization_strength, l1_ratio)
        y_observed.append(loss)
        print(
            f"Trial {i + 1:02d}/{iterations}: "
            f"a={steepness:.3f}, lambda={regularization_strength:.3e}, "
            f"l1_ratio={l1_ratio:.3f}, CV loss={loss:.4f}"
        )

    y_observed = np.asarray(y_observed, dtype=float)

    while len(y_observed) < iterations:
        gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=np.ones(3), nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=1,
        )
        gp.fit(X_observed, y_observed)

        candidates = rng.uniform(0.0, 1.0, size=(5000, 3))
        acquisition = expected_improvement(candidates, gp, np.min(y_observed))
        x_next = candidates[np.argmax(acquisition)]

        steepness, regularization_strength, l1_ratio = decode_search_point(x_next)
        loss = objective(steepness, regularization_strength, l1_ratio)

        X_observed = np.vstack([X_observed, x_next])
        y_observed = np.append(y_observed, loss)

        print(
            f"Trial {len(y_observed):02d}/{iterations}: "
            f"a={steepness:.3f}, lambda={regularization_strength:.3e}, "
            f"l1_ratio={l1_ratio:.3f}, CV loss={loss:.4f}"
        )

    rows = []
    for i, (x, loss) in enumerate(zip(X_observed, y_observed)):
        steepness, regularization_strength, l1_ratio = decode_search_point(x)
        rows.append(
            {
                "trial": i + 1,
                "steepness": steepness,
                "regularization_strength": regularization_strength,
                "l1_ratio": l1_ratio,
                "cv_cross_entropy": loss,
            }
        )

    return pd.DataFrame(rows)


def plot_results(
    optimization_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    classes: list[str],
    outfile: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(
        optimization_df.trial,
        optimization_df.cv_cross_entropy,
        marker="o",
        linewidth=1,
    )
    best_so_far = optimization_df.cv_cross_entropy.cummin()
    ax1.plot(optimization_df.trial, best_so_far, linewidth=2, label="Best so far")
    ax1.set_xlabel("Bayesian optimization trial")
    ax1.set_ylabel("Mean CV cross-entropy")
    ax1.set_title("Hyperparameter optimization")
    ax1.legend()

    cm = confusion_matrix(
        predictions_df.true_class,
        predictions_df.predicted_class,
        labels=np.arange(len(classes)),
    )
    image = ax2.imshow(cm)
    fig.colorbar(image, ax=ax2, fraction=0.046, pad=0.04)

    ax2.set_xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    ax2.set_yticks(np.arange(len(classes)), classes)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Repeated CV confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def main() -> None:
    # Load Snakemake or argparse arguments
    snakemake = globals().get("snakemake", None)
    cfg = parse_config(Config, snakemake)

    # These models are tiny. One CPU thread avoids PyTorch thread overhead.
    torch.set_num_threads(1)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print("Loading tree and trait data...")
    tree = skbio.read(cfg.input_tree, format="newick", into=skbio.TreeNode)
    trait_df = pd.read_csv(cfg.input_trait_data)

    if cfg.trait_column not in trait_df.columns:
        raise ValueError(f"Trait column not found: {cfg.trait_column}")

    if trait_df.worms_genus.duplicated().any():
        raise ValueError("Trait table contains duplicate worms_genus rows")

    trait_series = trait_df.set_index("worms_genus")[cfg.trait_column].dropna()

    tree_tip_names = [tip.name for tip in tree.tips()]
    if any(name is None for name in tree_tip_names):
        raise ValueError("All tree tips must have names")
    if len(tree_tip_names) != len(set(tree_tip_names)):
        raise ValueError("Tree tip names must be unique")

    taxa = np.array([name for name in tree_tip_names if name in trait_series.index], dtype=str)
    if len(taxa) < cfg.k_folds:
        raise ValueError("Not enough genera shared between tree and trait table for k-fold CV")

    missing_from_traits = sorted(set(tree_tip_names) - set(trait_series.index))
    print(f"Genera used for modelling: {len(taxa)}")
    print(f"Tree tips without this trait (including outgroups): {len(missing_from_traits)}")

    class_names = sorted(trait_series.loc[taxa].unique())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y = np.array([class_to_idx[trait_series.loc[name]] for name in taxa], dtype=int)

    print(f"Trait: {cfg.trait_column}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    print("Creating phylogenetic influence matrix...")
    influence, branch_lengths = create_influence_matrix(tree, taxa.tolist())

    # Every hyperparameter trial sees exactly the same repeated CV splits.
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.k_folds,
        n_repeats=cfg.repeats,
        random_state=cfg.seed,
    )
    splits = list(cv.split(taxa, y))

    def objective(steepness: float, regularization_strength: float, l1_ratio: float) -> float:
        loss, _ = evaluate_parameters(
            steepness=steepness,
            regularization_strength=regularization_strength,
            l1_ratio=l1_ratio,
            influence=influence,
            branch_lengths=branch_lengths,
            y=y,
            taxa=taxa,
            splits=splits,
            n_classes=len(class_names),
            cfg=cfg,
        )
        return loss

    print("\nStarting Bayesian optimization...")
    optimization_df = bayesian_optimization(
        objective=objective,
        iterations=cfg.bo_iterations,
        seed=cfg.seed,
    )

    best_row = optimization_df.loc[optimization_df.cv_cross_entropy.idxmin()]
    best_steepness = float(best_row.steepness)
    best_regularization = float(best_row.regularization_strength)
    best_l1_ratio = float(best_row.l1_ratio)

    print("\nEvaluating best parameters...")
    best_loss, predictions_df = evaluate_parameters(
        steepness=best_steepness,
        regularization_strength=best_regularization,
        l1_ratio=best_l1_ratio,
        influence=influence,
        branch_lengths=branch_lengths,
        y=y,
        taxa=taxa,
        splits=splits,
        n_classes=len(class_names),
        cfg=cfg,
        return_predictions=True,
    )
    assert predictions_df is not None

    predictions_df["true_label"] = predictions_df.true_class.map(dict(enumerate(class_names)))
    predictions_df["predicted_label"] = predictions_df.predicted_class.map(
        dict(enumerate(class_names))
    )

    accuracy = accuracy_score(predictions_df.true_class, predictions_df.predicted_class)
    macro_f1 = f1_score(
        predictions_df.true_class,
        predictions_df.predicted_class,
        average="macro",
    )

    for path in (
        cfg.output_optimization,
        cfg.output_predictions,
        cfg.output_summary,
        cfg.output_plot,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    optimization_df.to_csv(cfg.output_optimization, index=False)
    predictions_df.to_csv(cfg.output_predictions, index=False)
    plot_results(optimization_df, predictions_df, class_names, cfg.output_plot)

    summary = (
        f"Trait: {cfg.trait_column}\n"
        f"Genera: {len(taxa)}\n"
        f"Classes: {', '.join(class_names)}\n"
        f"CV: {cfg.repeats} repeats x {cfg.k_folds} folds\n"
        f"Bayesian optimization trials: {cfg.bo_iterations}\n\n"
        f"Best steepness (a): {best_steepness:.6f}\n"
        f"Best regularization strength (lambda): {best_regularization:.6e}\n"
        f"Best L1 ratio: {best_l1_ratio:.6f}\n\n"
        f"Mean CV cross-entropy: {best_loss:.6f}\n"
        f"Repeated-CV accuracy: {accuracy:.6f}\n"
        f"Repeated-CV macro F1: {macro_f1:.6f}\n"
    )
    Path(cfg.output_summary).write_text(summary)

    print("\n" + summary)
    print(f"Saved optimization history to: {cfg.output_optimization}")
    print(f"Saved CV predictions to: {cfg.output_predictions}")
    print(f"Saved summary to: {cfg.output_summary}")
    print(f"Saved result plot to: {cfg.output_plot}")


if __name__ == "__main__":
    main()
