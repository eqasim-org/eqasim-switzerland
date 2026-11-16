"""Descriptive density strata use recipient residence, never donor residence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series
from synthesis.population.matching.V3.inputs import scenario_mask

PURPOSES = ("work", "education", "shop", "leisure", "other")


def prepare(output, mappings, roster, persons, counts, chains, const, day, bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 1 or not len(bounds) or not np.isfinite(bounds).all() or (bounds <= 0).any() or (np.diff(bounds) <= 0).any():
        raise ValueError("Density boundaries must be finite, positive and increasing")
    labels = [f"<{bounds[0]:g}"] + [f"{a:g}–<{b:g}" for a, b in zip(bounds[:-1], bounds[1:])] + [f"≥{bounds[-1]:g}", "Unknown"]
    area = np.pi * (const.POPULATION_DENSITY_RADIUS / 1000.) ** 2
    def band(frame):
        raw = pd.to_numeric(frame.get("population_density", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        return np.where(np.isfinite(raw) & (raw >= 0), np.digitize(raw / area, bounds), len(labels) - 1)
    eligible = roster.age.ge(const.MZ_AGE_THRESHOLD) if const.census == "statpop" else roster.age_class.gt(0)
    common = pd.Index(roster.loc[eligible, "person_id"])
    for mapping in mappings.values():
        common = common.intersection(mapping.person_id, sort=False)
    population = roster.loc[roster.person_id.isin(common)].copy()
    population["density_bin"] = band(population)
    weight = pd.to_numeric(persons.person_weight, errors="coerce")
    selected = scenario_mask(persons, day) & ~coerce_boolean_series(persons.is_crossing_the_border, name="is_crossing_the_border")
    selected &= persons.age.ge(const.MZ_AGE_THRESHOLD) & np.isfinite(weight) & weight.gt(0)
    survey = persons.loc[selected].copy()
    survey["density_bin"] = band(survey)
    survey["weight"] = weight.loc[selected]
    survey["mz_person_id"] = survey.person_id
    datasets = {"Survey": survey}
    for name, mapping in mappings.items():
        frame = population.merge(mapping[["person_id", "mz_person_id"]], on="person_id", validate="one_to_one")
        frame["weight"] = 1.
        datasets[name] = frame
    stats, frequencies, coverage = [], [], []
    for dataset, frame in datasets.items():
        for number, group in frame.groupby("density_bin", sort=True):
            w = group.groupby("mz_person_id").weight.sum()
            if not len(w):
                continue
            n = counts.reindex(w.index, fill_value=0)[list(PURPOSES)]
            means = n.mul(w, axis=0).sum() / w.sum()
            for purpose in PURPOSES:
                stats.append(dict(dataset=dataset, density_bin=int(number), density_label=labels[number], purpose=purpose,
                    mean_activities=means[purpose], share_percent=100 * means[purpose] / means.sum() if means.sum() > 0 else np.nan,
                    people_with_activity_percent=float((n[purpose].gt(0) * w).sum() / w.sum() * 100)))
            matched_chains = chains.reindex(w.index)
            if matched_chains.isna().any():
                raise ValueError("Density comparison has missing donor chains")
            shares = w.groupby(matched_chains.to_numpy()).sum() / w.sum()
            frequencies.extend(dict(dataset=dataset, density_bin=int(number), density_label=labels[number], chain=chain, share=share) for chain, share in shares.items())
            coverage.append(dict(dataset=dataset, density_bin=int(number), density_label=labels[number], people=len(group),
                weight_sum=float(group.weight.sum()), distinct_donors=len(w), effective_people=float(group.weight.sum() ** 2 / (group.weight ** 2).sum())))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("density_statistics", stats), ("density_chains", frequencies), ("density_coverage", coverage)):
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    (output / "density_manifest.json").write_text(json.dumps({"bounds_per_km2": bounds.tolist(), "labels": labels,
        "algorithms": list(mappings), "common_recipients": len(common),
        "definition": "Resident count within the configured circular density radius divided by circle area in km²; population strata use recipient residence, survey strata use respondent residence."}, indent=2))


def render(output, save):
    import matplotlib.pyplot as plt
    output = Path(output)
    info = json.loads((output / "density_manifest.json").read_text())
    stats = pd.read_csv(output / "density_statistics.csv")
    chains = pd.read_csv(output / "density_chains.csv")
    datasets = ["Survey", *info["algorithms"]]
    colors = {"Survey": "#333333", "V1": "#dd8452", "V2": "#55a868", "V3": "#4c72b0"}
    active = sorted(stats.density_bin.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, purpose in zip(axes.flat, PURPOSES):
        for dataset in datasets:
            rows = stats.loc[stats.dataset.eq(dataset) & stats.purpose.eq(purpose)].set_index("density_bin").reindex(active)
            ax.plot(np.arange(len(active)), rows.mean_activities, marker="o", color=colors[dataset], label=dataset)
        ax.set(title=purpose.replace("shop", "Shopping").capitalize(), ylabel="Activities per resident/day")
        ax.set_xticks(np.arange(len(active)), [info["labels"][i] for i in active], rotation=30, ha="right")
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    save(fig, "activity_rates_by_density", "National comparison by residential density (residents/km²). The population curves use the same recipients retained by all algorithms. Each person is classified using their own residential density; survey curves use respondent density and survey weights. Zero-activity days remain in the denominator. These associations also reflect demographic and household differences; they are not causal effects of density.")
    ranking = pd.read_csv(output / "chains.csv")
    ranking = ranking.loc[ranking.scope.eq("common") & ranking.dataset.eq("Survey")].sort_values("share", ascending=False).head(15).chain.tolist()
    combinations = [(dataset, density) for density in active for dataset in datasets]
    matrix = np.full((len(combinations), len(ranking)), np.nan)
    for i, (dataset, density) in enumerate(combinations):
        values = chains.loc[chains.dataset.eq(dataset) & chains.density_bin.eq(density)].set_index("chain").share
        if len(values):
            matrix[i] = values.reindex(ranking, fill_value=0).to_numpy() * 100
    fig, ax = plt.subplots(figsize=(15, max(6, len(combinations) * .36)))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_xticks(np.arange(len(ranking)), ranking, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(combinations)), [f'{info["labels"][density]} | {dataset}' for dataset, density in combinations], fontsize=9)
    fig.colorbar(image, ax=ax, label="People with this chain (%)")
    save(fig, "activity_chains_by_density", "Each row is a density group and matching method. Columns are the national survey's 15 most frequent chains. Colour shows the fraction of people in that density group with each chain; the unshown tail remains in the denominator. Grey/masked rows have no observations. Population results use the common matched cohort; survey uses person weights. The CSV contains all chains, not only these 15.")
