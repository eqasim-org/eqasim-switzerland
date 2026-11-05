"""Weighted diary summaries and report rendering for population V1/V3 assignments.

Re-render saved tables with:
python -m analysis.matching.population_comparison /path/to/analysis/matching/v1_v3
"""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series
from synthesis.population.matching.V3.inputs import scenario_mask
from . import distance_maps

PURPOSES = ("work", "education", "shop", "leisure", "other")
LABELS = {"work": "Work", "education": "Education", "shop": "Shopping", "leisure": "Leisure", "other": "Other"}
DATASETS = ("Survey", "V1", "V3")
CANTONS = (("geneva", 25, "Geneva"), ("zurich", 1, "Zürich"), ("vaud", 22, "Vaud"))


def activity_episodes(trips):
    """Observed activity starts/ends; initial/final episodes are boundary-censored."""
    if trips.duplicated(["person_id", "trip_id"]).any():
        raise ValueError("Survey trips have duplicate person/trip identifiers")
    t = trips.sort_values(["person_id", "trip_id"]).copy()
    for column in ("departure_time", "arrival_time"):
        t[column] = pd.to_numeric(t[column], errors="coerce")
    next_departure = t.groupby("person_id").departure_time.shift(-1)
    final = t.groupby("person_id").cumcount(ascending=False).eq(0)
    episodes = pd.DataFrame({"person_id": t.person_id, "purpose": t.purpose,
        "start_h": t.arrival_time / 3600, "end_h": next_departure / 3600,
        "initial": False, "final": final})
    first = t.groupby("person_id", sort=False).head(1)
    initial = pd.DataFrame({"person_id": first.person_id, "purpose": first.origin_purpose,
        "start_h": 0., "end_h": first.departure_time / 3600, "initial": True, "final": False})
    episodes = pd.concat([initial, episodes], ignore_index=True)
    episodes["purpose"] = episodes.purpose.astype(str).str.replace(r"_secondary$|_main$", "", regex=True).replace({"shopping": "shop"})
    episodes["duration_h"] = episodes.end_h - episodes.start_h
    return episodes


def generate(output, mappings, roster, persons, trips, chains, const, day, *, cantons=None, distance_map_options=None):
    """Reuse diary summaries for national and home-canton comparisons."""
    episodes = activity_episodes(trips)
    counts = pd.crosstab(episodes.person_id, episodes.purpose).reindex(columns=PURPOSES, fill_value=0)
    counts = counts.reindex(pd.Index(persons.person_id).union([-1]), fill_value=0)
    if chains.person_id.duplicated().any():
        raise ValueError("Comparison requires unique survey chain identifiers")
    chain_values = chains.set_index("person_id").activity_chain.copy()
    chain_values.loc[-1] = "home"
    prepared = (episodes, counts, chain_values)
    map_options = dict(distance_map_options) if distance_map_options is not None else None
    canton_grid_size = map_options.pop("canton_grid_size", map_options.get("grid_size", 5000.)) if map_options is not None else None
    regional_reports = []
    if cantons is not None:
        known = set(pd.to_numeric(cantons.canton_id, errors="raise"))
        if not {code for _, code, _ in CANTONS}.issubset(known):
            raise ValueError("data.spatial.cantons is missing Geneva, Zürich or Vaud")
        for frame in (roster, persons):
            if "canton_id" not in frame:
                raise ValueError("Regional comparisons require upstream home canton_id")
        for slug, code, label in CANTONS:
            if map_options is not None:
                local_options = {**map_options, "grid_size": canton_grid_size}
                distance_maps.prepare(Path(output) / slug, mappings, roster, persons, trips, const, day,
                    cantons=cantons, canton_id=code, region=label, **local_options)
            _generate_one(Path(output) / slug, mappings, roster, persons, const, day,
                prepared, canton_id=code, region=label, has_distance_maps=map_options is not None)
            regional_reports.append({"directory": slug, "label": label, "canton_id": code})
    if distance_map_options is not None:
        distance_maps.prepare(output, mappings, roster, persons, trips, const, day,
            cantons=cantons, **map_options)
    return _generate_one(output, mappings, roster, persons, const, day,
        prepared, regional_reports=regional_reports, has_distance_maps=distance_map_options is not None)


def _generate_one(output, mappings, roster, persons, const, day, prepared, *,
                  canton_id=None, region="Switzerland", regional_reports=(), has_distance_maps=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    eligible = roster.age.ge(const.MZ_AGE_THRESHOLD) if const.census == "statpop" else roster.age_class.gt(0)
    if canton_id is not None:
        eligible &= pd.to_numeric(roster.canton_id, errors="coerce").eq(canton_id)
    recipients = pd.Index(roster.loc[eligible, "person_id"])
    if recipients.has_duplicates or persons.person_id.duplicated().any():
        raise ValueError("Comparison requires unique population/survey/chain identifiers")
    reference = scenario_mask(persons, day) & ~coerce_boolean_series(persons.is_crossing_the_border, name="is_crossing_the_border")
    reference &= persons.age.ge(const.MZ_AGE_THRESHOLD)
    survey_weights = pd.to_numeric(persons.person_weight, errors="coerce")
    reference &= np.isfinite(survey_weights) & survey_weights.gt(0)
    if canton_id is not None:
        reference &= pd.to_numeric(persons.canton_id, errors="coerce").eq(canton_id)
    survey = pd.Series(survey_weights.loc[reference].to_numpy(), index=persons.loc[reference, "person_id"])
    if survey.empty and canton_id is None:
        raise ValueError("No eligible survey respondents for the comparison")
    clean = {}
    for name, frame in mappings.items():
        if frame.person_id.duplicated().any() or not frame.person_id.isin(roster.person_id).all():
            raise ValueError(f"{name} changed the population identity set")
        selected = frame.loc[frame.person_id.isin(recipients)].copy()
        clean[name] = selected
    common = recipients
    for frame in clean.values():
        common = common.intersection(frame.person_id, sort=False)
    coverage = pd.DataFrame([{"algorithm": name, "input_eligible": len(recipients),
        "retained": len(frame), "removed": len(recipients) - len(frame),
        "no_donor_home": int(frame.mz_person_id.eq(-1).sum()), "common_recipients": len(common)} for name, frame in clean.items()])
    coverage.to_csv(output / "coverage.csv", index=False)
    episodes, counts, chain_values = prepared
    chain_rows, count_rows, time_rows, rate_rows = [], [], [], []
    for scope in ("full", "common"):
        weights = {"Survey": survey}
        for name, frame in clean.items():
            if scope == "common":
                frame = frame.loc[frame.person_id.isin(common)]
            weights[name] = frame.mz_person_id.value_counts(sort=False).astype(float)
        for dataset, w in weights.items():
            if w.empty:
                continue
            if (w.index < -1).any() or not w.index.isin(counts.index).all():
                raise ValueError(f"{dataset} has unknown donor identifiers")
            chain = chain_values.reindex(w.index)
            if chain.isna().any():
                raise ValueError(f"{dataset} has missing activity chains")
            shares = w.groupby(chain.to_numpy()).sum() / w.sum()
            chain_rows.extend({"scope": scope, "dataset": dataset, "chain": key, "share": value} for key, value in shares.items())
            for purpose in PURPOSES:
                n = counts.loc[w.index, purpose].to_numpy()
                probabilities = np.bincount(np.minimum(n, 15).astype(int), weights=w.to_numpy(), minlength=16) / w.sum()
                count_rows.extend({"scope": scope, "dataset": dataset, "purpose": purpose, "count": k, "probability": value} for k, value in enumerate(probabilities))
                rate_rows.append({"scope": scope, "dataset": dataset, "purpose": purpose,
                    "activities_per_100_people": float(np.dot(n, w.to_numpy()) / w.sum() * 100)})
            event_weights = episodes.person_id.map(w).fillna(0).to_numpy()
            for purpose in PURPOSES:
                purpose_mask = episodes.purpose.eq(purpose).to_numpy()
                for metric in ("start_h", "end_h", "duration_h"):
                    allowed = purpose_mask.copy()
                    if metric in ("start_h", "duration_h"):
                        allowed &= ~episodes.initial.to_numpy()
                    if metric in ("end_h", "duration_h"):
                        allowed &= ~episodes.final.to_numpy()
                    values = episodes[metric].to_numpy()
                    valid = allowed & np.isfinite(values) & (values >= 0) & (values <= 30)
                    hist, edges = np.histogram(values[valid], bins=np.linspace(0, 30, 61), weights=event_weights[valid])
                    mass = hist.sum()
                    density = hist / mass / np.diff(edges) if mass > 0 else np.zeros_like(hist)
                    excluded = float(event_weights[allowed & ~valid].sum())
                    time_rows.extend({"scope": scope, "dataset": dataset, "purpose": purpose, "metric": metric,
                        "left_h": edges[k], "right_h": edges[k + 1], "density": value,
                        "valid_event_weight": mass, "invalid_event_weight": excluded} for k, value in enumerate(density))
    for name, rows in (("chains", chain_rows), ("activity_counts", count_rows), ("activity_times", time_rows), ("purpose_rates", rate_rows)):
        columns = {"chains": ["scope", "dataset", "chain", "share"],
            "activity_counts": ["scope", "dataset", "purpose", "count", "probability"],
            "activity_times": ["scope", "dataset", "purpose", "metric", "left_h", "right_h", "density", "valid_event_weight", "invalid_event_weight"],
            "purpose_rates": ["scope", "dataset", "purpose", "activities_per_100_people"]}
        pd.DataFrame(rows, columns=columns[name]).to_csv(output / f"{name}.csv", index=False)
    manifest = {"scenario_day": day, "survey_respondents": len(survey), "eligible_population": len(recipients),
        "common_population": len(common), "survey_weight_sum": float(survey.sum()),
        "region": region, "canton_id": canton_id, "regional_reports": list(regional_reports),
        "distance_maps": has_distance_maps,
        "geography": "Population weights use each recipient's home canton, never the assigned donor's canton. Survey weights use the respondent's home canton. Upstream canton IDs are checked against data.spatial.cantons.",
        "scope": "Domestic age-eligible scenario-day survey; synthetic age-eligible retained residents.",
        "timing": "Starts and ends in hours; durations use completed internal episodes only; 0–30 h range.",
        "interpretation": "Descriptive assigned-diary comparison, before downstream trip recoding, remote-work changes or location assignment; no independent accuracy claim."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return render_report(output, include_regions=False)


def render_report(output, *, include_regions=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    regional_reports = manifest.get("regional_reports", [])
    if include_regions:
        for region in regional_reports:
            render_report(output / region["directory"])
    tables = {name: pd.read_csv(output / f"{name}.csv") for name in ("coverage", "chains", "activity_counts", "activity_times", "purpose_rates")}
    colors = {"Survey": "#333333", "V1": "#dd8452", "V3": "#4c72b0"}
    figures = []

    def save(fig, name, explanation):
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        figures.append((name, explanation))

    if manifest.get("distance_maps", False):
        distance_maps.render(output, save)
        if (output / "distance_distributions.csv").exists():
            distance_maps.render_distributions(output, save)

    for scope in ("full", "common"):
        chains = tables["chains"].query("scope == @scope")
        ranking = chains.loc[chains.dataset.eq("Survey")].sort_values(["share", "chain"], ascending=[False, True]).head(15).chain.tolist()
        ranking_label = "survey"
        if not ranking:
            ranking = chains.groupby("chain").share.sum().sort_values(ascending=False).head(15).index.tolist()
            ranking_label = "population (no local survey)"
        fig, ax = plt.subplots(figsize=(12, 8))
        positions = np.arange(len(ranking) + 1)
        for i, dataset in enumerate(DATASETS):
            values = chains.loc[chains.dataset.eq(dataset)].set_index("chain").share
            shown = values.reindex(ranking, fill_value=0).to_numpy()
            values_plot = np.r_[shown, max(0, 1 - shown.sum()) if len(values) else 0]
            ax.barh(positions + (i - 1) * .25, values_plot * 100, height=.24, label=dataset, color=colors[dataset])
        ax.set_yticks(positions, ranking + ["All remaining chains"])
        ax.invert_yaxis()
        ax.set_xlabel("Share of people (%)")
        ax.set_title(f"15 most frequent {ranking_label} activity chains — {manifest.get('region', 'Switzerland')} — {scope} population")
        ax.legend()
        save(fig, f"{scope}_chains", "Chains describe the sequence of daily purposes. The same 15 chains, ranked by local survey-weighted frequency when available, are shown for all three datasets; without a local survey they are ranked by pooled population shares. The last row contains the rest. Home-only days remain in the denominator.")
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig.suptitle(f"Daily activity counts — {manifest.get('region', 'Switzerland')} — {scope} population")
        for ax, purpose in zip(axes.flat, PURPOSES):
            for dataset in DATASETS:
                rows = tables["activity_counts"].query("scope == @scope and purpose == @purpose and dataset == @dataset")
                ax.plot(rows["count"], rows.probability, marker=".", label=dataset, color=colors[dataset])
            ax.set(title=LABELS[purpose], xlabel="Activities per person/day (15 = 15+)", ylabel="Probability")
        axes.flat[-1].axis("off")
        axes.flat[0].legend()
        save(fig, f"{scope}_counts", "Each curve gives the probability of having zero, one, two, or more activities of that purpose. Non-travellers contribute zero activities. These are discrete probabilities, including a 15-or-more tail, rather than continuous densities.")
        for metric, label in (("start_h", "Activity start time"), ("end_h", "Activity end time"), ("duration_h", "Completed activity duration")):
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            fig.suptitle(f"{label} — {manifest.get('region', 'Switzerland')} — {scope} population")
            for ax, purpose in zip(axes.flat, PURPOSES):
                for dataset in DATASETS:
                    rows = tables["activity_times"].query("scope == @scope and purpose == @purpose and metric == @metric and dataset == @dataset")
                    ax.step((rows.left_h + rows.right_h) / 2, rows.density, where="mid", label=dataset, color=colors[dataset])
                ax.set(title=LABELS[purpose], xlabel=f"{label} (hours)", ylabel="Density (per hour)", xlim=(0, 30))
            axes.flat[-1].axis("off")
            axes.flat[0].legend()
            save(fig, f"{scope}_{metric}", f"{label}: curves use half-hour bins and integrate to one when events are available. Starts exclude the artificial start of the initial activity; ends exclude the unknown end of the final activity. Durations include only completed internal activities. Missing, negative and out-of-range observations are excluded and counted in the CSV tables; empty curves mean no valid observations.")
    intro = ("This report compares the diaries assigned by V1 and V3 with scenario-day Microcensus diaries. Survey results use person weights; population results weight each donor by the number of recipients assigned to that donor. Counts therefore retain non-travellers without expanding the full population activity table. "
        "The full-population figures use every age-eligible person retained by each method, including explicitly home-only residents. Methods can remove different people: coverage is shown below. The common-population figures repeat the comparison on exactly the same retained recipients, separating diary differences from coverage differences. "
        "These are assigned-diary distributions before downstream adjustments, not final simulated activity locations or observed population truth. Similarity to the survey alone does not establish which method predicts behaviour better.")
    body = ["<!doctype html><meta charset='utf-8'><title>V1–V3 diary comparison</title>",
        "<style>body{max-width:1250px;margin:35px auto;font:17px/1.5 sans-serif;padding:15px}img{width:100%}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ccc}</style>",
        f"<h1>V1–V3 population diary comparison: {html.escape(manifest.get('region', 'Switzerland'))}</h1>", f"<p>{html.escape(intro)}</p>",
        f"<p>Scenario day: {html.escape(str(manifest['scenario_day']))}. Survey respondents: {manifest['survey_respondents']:,}.</p>",
        "<h2>Coverage</h2>", tables["coverage"].to_html(index=False),
        "<h2>Activity rates</h2><p>Expected activities per 100 people per day. These rates complement the distributions below.</p>",
        tables["purpose_rates"].to_html(index=False, float_format=lambda x: f"{x:.2f}")]
    if regional_reports:
        navigation = " | ".join(f"<a href='{html.escape(r['directory'])}/report.html'>{html.escape(r['label'])}</a>" for r in regional_reports)
        body.insert(4, f"<p>Canton reports: {navigation}</p>")
    if manifest.get("canton_id") is not None:
        body.insert(4, "<p><a href='../report.html'>Switzerland and other cantons</a></p>"
            "<p>This report selects synthetic residents by their own home canton. Their assigned donors may live elsewhere. The survey comparison includes only respondents living in this canton. Each distribution is normalized within this canton and dataset; the top chains are ranked using this canton's survey weights. Empty curves mean no observations; zero survey respondents means no local survey reference is available.</p>")
    for name, explanation in figures:
        body += [f"<h2>{html.escape(name.replace('_', ' '))}</h2>", f"<p>{html.escape(explanation)}</p>", f"<img src='{name}.png' alt='{html.escape(name)}'>"]
    body.append("<p>All values are exported as CSV files alongside PNG/PDF figures. Re-render this report from these tables without rerunning either matcher using the population_comparison module.</p>")
    (output / "report.html").write_text("\n".join(body))
    return str(output / "report.html")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-render saved V1/V3 comparison tables; no matching or simulation.")
    parser.add_argument("report_directory", type=Path)
    print(render_report(parser.parse_args().report_directory))
