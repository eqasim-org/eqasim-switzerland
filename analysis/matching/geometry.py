"""Out-of-sample neighbourhood quality, projection fidelity and seed stability."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from threadpoolctl import threadpool_limits

from .metrics import clustered_interval
from .reporting import save_figure


def geometry_diagnostics(source, output, manifest, seed, maximum, method):
    rng, rows, stability = np.random.RandomState(seed), [], []
    projection = None
    for fold in range(manifest["outer_folds"]):
        reference = None
        for repeat in range(manifest["model_repeats"]):
            directory = source / f"repeat_{repeat}_fold_{fold}"
            donors = pd.read_csv(directory / "donors_embeddings.csv")
            test = pd.read_csv(directory / "heldout_embeddings.csv")
            a = donors.filter(regex=r"^emb_").to_numpy()
            b = test.filter(regex=r"^emb_").to_numpy()
            actual = np.load(directory / "heldout_outcomes.npy")
            donor_outcomes = np.load(directory / "donors_outcomes.npy")
            learned = np.load(directory / "learned_neighbors.npy")
            raw = np.load(directory / "raw_neighbors.npy")
            centered = b - b.mean(axis=0)
            variance = np.linalg.svd(centered, full_matrices=False, compute_uv=False) ** 2
            mass = variance / max(variance.sum(), 1e-12)
            effective_rank = (np.exp(-np.sum(mass[mass > 0] * np.log(mass[mass > 0])))
                              if variance.sum() > 1e-12 else 0.)
            test_rows = rng.choice(len(test), min(maximum, len(test)), replace=False)
            sampled = b[test_rows]
            projected = _project(sampled, method, seed)
            k = min(10, max((len(sampled)-1)//2, 1))
            fidelity = trustworthiness(sampled, projected, n_neighbors=k) if len(sampled) > 3 else np.nan
            pair_a, pair_b = _within_mask_pairs(test, rng, 3000)
            latent_distance = np.linalg.norm(b[pair_a] - b[pair_b], axis=1)
            behaviour_distance = np.linalg.norm(actual[pair_a] - actual[pair_b], axis=1)
            alignment = _spearman(latent_distance, behaviour_distance)
            lookup = pd.Series(np.arange(len(donors)), index=donors["mz_id"])
            for label, neighbor_ids in (("learned", learned), ("raw", raw)):
                for count in (5, 20):
                    distances = []
                    for i, ids in enumerate(neighbor_ids[:, :count]):
                        local = lookup.reindex(ids[ids >= 0]).dropna().to_numpy(dtype=int)
                        if len(local):
                            distances.append(np.linalg.norm(donor_outcomes[local] - actual[i], axis=1).mean())
                    rows.append(dict(fold=fold, repeat=repeat, space=label, k=count,
                        mean_neighbour_behaviour_distance=float(np.mean(distances)) if distances else np.nan,
                        neighbourhood_coverage=float(np.mean(neighbor_ids[:, 0] >= 0)),
                        latent_effective_rank=float(effective_rank) if label == "learned" else np.nan,
                        projection_trustworthiness=float(fidelity) if label == "learned" else np.nan,
                        within_mask_behaviour_spearman=alignment if label == "learned" else np.nan))
            if reference is None:
                reference = (test["mz_id"].to_numpy(), b, learned)
            else:
                if not np.array_equal(reference[0], test["mz_id"].to_numpy()):
                    raise ValueError("Seed stability requires identical outer test respondents")
                distance_reference = np.linalg.norm(reference[1][pair_a] - reference[1][pair_b], axis=1)
                stability.append(dict(fold=fold, repeat=repeat,
                    distance_spearman=_spearman(distance_reference, latent_distance),
                    neighbour_jaccard=float(np.mean([_jaccard(x, y) for x, y in zip(reference[2], learned)]))))
            if projection is None:
                projection = test.iloc[test_rows].copy()
                projection["projection_1"], projection["projection_2"] = projected[:, 0], projected[:, 1]
                projection.to_csv(output / "heldout_projection.csv", index=False)
                _projection_figure(projection, output, method)
                _behaviour_projection(projection, source, output)
                _projection_sensitivity(sampled, projection, projected, method, output, seed)
    _neighbour_figure(pd.DataFrame(rows), output)
    return pd.DataFrame(rows), pd.DataFrame(stability)


def _project(sampled, method, seed, perplexity=30.):
    # Small visualizations must not spawn large nested BLAS/OpenMP thread pools.
    with threadpool_limits(limits=1):
        if method == "tsne" and len(sampled) > 4:
            return TSNE(n_components=2, init="pca", learning_rate="auto",
                        perplexity=min(perplexity, (len(sampled)-1)/3), random_state=seed,
                        n_jobs=1).fit_transform(sampled)
        return PCA(n_components=2, random_state=seed).fit_transform(sampled)


def _behaviour_projection(projection, source, output):
    columns = ["mz_id", "algorithm", "repeat", "fold", "truth_trip_count", "truth_distance_km", "truth_work", "truth_education"]
    truth = pd.read_csv(source / "out_of_fold_scores.csv", usecols=columns)
    truth = truth.loc[truth.algorithm.eq("V2") & truth["repeat"].eq(0) & truth.fold.eq(0)]
    frame = projection.merge(truth, on="mz_id", validate="one_to_one")
    frame.to_csv(output / "heldout_projection_with_behaviour.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, feature, label in zip(axes.ravel(),
            ("truth_trip_count", "truth_distance_km", "truth_work", "truth_education"),
            ("Observed daily trips", "Observed daily distance (log1p km)", "Observed work day", "Observed education day")):
        color = frame[feature].to_numpy()
        if feature == "truth_distance_km":
            color = np.log1p(color)
        points = axis.scatter(frame.projection_1, frame.projection_2, c=color, s=9, alpha=.65, cmap="viridis")
        fig.colorbar(points, ax=axis, label=label)
        axis.set(xlabel="Projection coordinate 1", ylabel="Projection coordinate 2")
    save_figure(fig, output, "embedding_behaviour")


def _projection_sensitivity(sampled, frame, main, method, output, seed):
    views = [("PCA", _project(sampled, "pca", seed)), (method.upper() + " (main)", main)]
    for perplexity in (15., 50.):
        used = min(perplexity, (len(sampled)-1)/3)
        views.append((f"t-SNE perplexity {used:.1f}", _project(sampled, "tsne", seed, perplexity)))
    rows = []
    fig, axes = plt.subplots(1, len(views), figsize=(15, 3.5))
    k = min(10, max((len(sampled)-1)//2, 1))
    for axis, (label, points) in zip(axes, views):
        fidelity = trustworthiness(sampled, points, n_neighbors=k) if len(sampled) > 3 else np.nan
        rows.append(dict(view=label, seed=seed, respondents=len(sampled), neighbors=k, trustworthiness=fidelity))
        coords = frame[["mz_id"]].assign(projection_1=points[:, 0], projection_2=points[:, 1])
        coords.to_csv(output / f"projection_sensitivity_{len(rows)}.csv", index=False)
        axis.scatter(points[:, 0], points[:, 1], c=frame.age, s=7, alpha=.6, cmap="viridis")
        axis.set(title=f"{label}\nTrustworthiness {fidelity:.3f}", xlabel="Coordinate 1", ylabel="Coordinate 2")
    save_figure(fig, output, "projection_sensitivity")
    pd.DataFrame(rows).to_csv(output / "projection_sensitivity.csv", index=False)


def _neighbour_figure(frame, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, count in zip(axes, (5, 20)):
        for _, group in frame.loc[frame.k.eq(count)].groupby(["fold", "repeat"]):
            pair = group.set_index("space").reindex(["raw", "learned"])
            axis.plot([0, 1], pair.mean_neighbour_behaviour_distance, "o-", alpha=.6)
        axis.set(xticks=[0, 1], xticklabels=["Raw inputs", "Learned inputs"],
                 title=f"Up to {count} eligible neighbours", ylabel="Average standardized behaviour distance")
    save_figure(fig, output, "neighbour_behaviour")


def _within_mask_pairs(test, rng, maximum):
    groups = list(test.groupby(["employed", "is_student"]).indices.values())
    groups = [group for group in groups if len(group) >= 2]
    if not groups:
        return np.array([], dtype=int), np.array([], dtype=int)
    a, b = [], []
    for _ in range(maximum):
        group = groups[rng.randint(len(groups))]
        pair = rng.choice(group, 2, replace=False)
        a.append(pair[0]); b.append(pair[1])
    return np.asarray(a), np.asarray(b)


def _spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def _jaccard(a, b):
    a, b = set(a[a >= 0]), set(b[b >= 0])
    return len(a & b) / len(a | b) if a | b else np.nan


def _projection_figure(frame, output, method):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, feature in zip(axes, ("age", "income_class", "population_density")):
        color = frame[feature].to_numpy()
        if feature == "population_density":
            color = np.log1p(color)
        if not np.isfinite(color).any():
            axis.text(.5, .5, feature + " unavailable", ha="center", transform=axis.transAxes)
            continue
        scatter = axis.scatter(frame["projection_1"], frame["projection_2"], c=color, cmap="viridis", s=9, alpha=.6)
        fig.colorbar(scatter, ax=axis, label=feature + (" (log1p)" if feature == "population_density" else ""))
        axis.set(xlabel=method.upper() + " 1", ylabel=method.upper() + " 2")
    save_figure(fig, output, "heldout_representation")


def permutation_summary(raw, respondents, bootstrap, seed):
    # Multiple permutations/model fits are repeated measurements, not independent people.
    average = raw.groupby(["mz_id", "group"], as_index=False)["energy_increase"].mean()
    average = average.merge(respondents, on="mz_id", validate="many_to_one")
    rows, rng = [], np.random.RandomState(seed)
    for weighting in ("survey_weight", "target_weight"):
        for name, group in average.groupby("group"):
            estimate, low, high, n, ess = clustered_interval(
                group["energy_increase"], group[weighting], group["household_group"], rng, bootstrap)
            rows.append(dict(feature_group=name, weighting=weighting, energy_increase=estimate,
                             low=low, high=high, households=n, effective_n=ess))
    return pd.DataFrame(rows)


def importance_figure(frame, output):
    group = frame.loc[frame["weighting"] == "survey_weight"].sort_values("energy_increase")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(group["energy_increase"], np.arange(len(group)),
                  xerr=np.maximum(np.vstack([group["energy_increase"] - group["low"], group["high"] - group["energy_increase"]]), 0),
                  fmt="o", color="#0072B2", capsize=3)
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_yticks(np.arange(len(group)), [name.replace("_", " ").capitalize() for name in group["feature_group"]])
    axis.set_xlabel("Held-out energy-score increase after grouped permutation")
    save_figure(fig, output, "predictive_feature_contributions")
