"""Weighted diary summaries and report rendering for population diary assignments.

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
from . import distance_maps, density_comparison

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


def generate(output, mappings, roster, persons, trips, chains, const, day, *, cantons=None, distance_map_options=None, report_metadata=None, density_bounds=None):
    """Reuse diary summaries for national and home-canton comparisons."""
    episodes = activity_episodes(trips)
    counts = pd.crosstab(episodes.person_id, episodes.purpose).reindex(columns=PURPOSES, fill_value=0)
    counts = counts.reindex(pd.Index(persons.person_id).union([-1]), fill_value=0)
    if chains.person_id.duplicated().any():
        raise ValueError("Comparison requires unique survey chain identifiers")
    chain_values = chains.set_index("person_id").activity_chain.copy()
    chain_values.loc[-1] = "home"
    home_counts = episodes.loc[episodes.purpose.eq("home")].groupby("person_id").size()
    home_counts = home_counts.reindex(counts.index, fill_value=0)
    no_trips = ~counts.index.isin(episodes.person_id)
    home_counts.loc[no_trips] = 1
    prepared = (episodes, counts, chain_values, home_counts)
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
                prepared, canton_id=code, region=label, has_distance_maps=map_options is not None, report_metadata=report_metadata)
            regional_reports.append({"directory": slug, "label": label, "canton_id": code})
    if distance_map_options is not None:
        distance_maps.prepare(output, mappings, roster, persons, trips, const, day,
            cantons=cantons, **map_options)
    if density_bounds is not None:
        density_comparison.prepare(output, mappings, roster, persons, counts, chain_values, const, day, density_bounds)
    return _generate_one(output, mappings, roster, persons, const, day,
        prepared, regional_reports=regional_reports, has_distance_maps=distance_map_options is not None,
        report_metadata={**(report_metadata or {}), "density_analysis": density_bounds is not None})


def _generate_one(output, mappings, roster, persons, const, day, prepared, *,
                  canton_id=None, region="Switzerland", regional_reports=(), has_distance_maps=False, report_metadata=None):
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
    episodes, counts, chain_values, home_counts = prepared
    chain_rows, count_rows, time_rows, rate_rows, daily_rows = [], [], [], [], []
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
                    "activities_per_100_people": float(np.dot(n, w.to_numpy()) / w.sum() * 100),
                    "mean_activities_per_person": float(np.dot(n, w.to_numpy()) / w.sum()),
                    "people_with_activity_percent": float(np.dot(n > 0, w.to_numpy()) / w.sum() * 100)})
            home = home_counts.reindex(w.index, fill_value=1).to_numpy()
            nonhome = counts.loc[w.index, list(PURPOSES)].sum(axis=1).to_numpy()
            mean_home = float(np.dot(home, w.to_numpy()) / w.sum())
            mean_nonhome = float(np.dot(nonhome, w.to_numpy()) / w.sum())
            daily_rows.append(dict(scope=scope, dataset=dataset, mean_total_activities=mean_home + mean_nonhome,
                mean_home_activities=mean_home, mean_nonhome_activities=mean_nonhome,
                people_with_nonhome_activity_percent=float(np.dot(nonhome > 0, w.to_numpy()) / w.sum() * 100)))
            rate_rows.append(dict(scope=scope, dataset=dataset, purpose="home", activities_per_100_people=mean_home * 100,
                mean_activities_per_person=mean_home,
                people_with_activity_percent=float(np.dot(home > 0, w.to_numpy()) / w.sum() * 100)))
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
    pd.DataFrame(daily_rows, columns=["scope", "dataset", "mean_total_activities", "mean_home_activities",
        "mean_nonhome_activities", "people_with_nonhome_activity_percent"]).to_csv(output / "daily_activity_totals.csv", index=False)
    for name, rows in (("chains", chain_rows), ("activity_counts", count_rows), ("activity_times", time_rows), ("purpose_rates", rate_rows)):
        columns = {"chains": ["scope", "dataset", "chain", "share"],
            "activity_counts": ["scope", "dataset", "purpose", "count", "probability"],
            "activity_times": ["scope", "dataset", "purpose", "metric", "left_h", "right_h", "density", "valid_event_weight", "invalid_event_weight"],
            "purpose_rates": ["scope", "dataset", "purpose", "activities_per_100_people", "mean_activities_per_person", "people_with_activity_percent"]}
        table = pd.DataFrame(rows, columns=columns[name])
        if name == "purpose_rates":
            keys = [table.scope, table.dataset]
            total = table.mean_activities_per_person.groupby(keys).transform("sum")
            nonhome = table.mean_activities_per_person.where(table.purpose.ne("home"), 0).groupby(keys).transform("sum")
            table["share_of_nonhome_activities_percent"] = (100 * table.mean_activities_per_person / nonhome.where(nonhome.gt(0))).where(table.purpose.ne("home"))
            table["share_of_all_activities_percent"] = 100 * table.mean_activities_per_person / total.where(total.gt(0))
        table.to_csv(output / f"{name}.csv", index=False)
    manifest = {"scenario_day": day, "survey_respondents": len(survey), "eligible_population": len(recipients),
        "common_population": len(common), "survey_weight_sum": float(survey.sum()),
        "region": region, "canton_id": canton_id, "regional_reports": list(regional_reports),
        "distance_maps": has_distance_maps, "algorithms": list(mappings),
        "geography": "Population weights use each recipient's home canton, never the assigned donor's canton. Survey weights use the respondent's home canton. Upstream canton IDs are checked against data.spatial.cantons.",
        "scope": "Domestic age-eligible scenario-day survey; synthetic age-eligible retained residents.",
        "timing": "Starts and ends in hours; durations use completed internal episodes only; 0–30 h range.",
        "interpretation": "Descriptive assigned-diary comparison, before downstream trip recoding, remote-work changes or location assignment; no independent accuracy claim."}
    manifest.update(report_metadata or {})
    if "exclusions_by_region" in manifest:
        manifest["excluded_people"] = manifest["exclusions_by_region"].get(region, manifest["excluded_people"])
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
    datasets = ("Survey", *manifest.get("algorithms", ["V1", "V3"]))
    colors = {"Survey": "#333333", "V1": "#dd8452", "V2": "#55a868", "V3": "#4c72b0"}
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

    if manifest.get("density_analysis", False):
        density_comparison.render(output, save)

    if manifest.get("forced_home_only_excluded"):
        rates = tables["purpose_rates"]
        purposes = ["home", *PURPOSES]
        palette = ["#999999", "#4c72b0", "#dd8452", "#55a868", "#8172b3", "#937860"]
        for scope in ("common", "full"):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, metric, ylabel in zip(axes, ("mean_activities_per_person", "share_of_all_activities_percent"),
                                         ("Activities per resident/day", "Share of all activities (%)")):
                bottom = np.zeros(len(datasets))
                for purpose, color in zip(purposes, palette):
                    values = rates.loc[rates.scope.eq(scope) & rates.purpose.eq(purpose)].set_index("dataset")[metric].reindex(datasets).fillna(0).to_numpy()
                    ax.bar(datasets, values, bottom=bottom, label=LABELS.get(purpose, "Home"), color=color)
                    bottom += values
                ax.set_ylabel(ylabel)
            axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
            fig.suptitle(f"{manifest.get('region', 'Switzerland')} — average resident's day — {scope} population")
            save(fig, f"{scope}_average_day", "Left: the average number of activities per resident per day, including repeated home visits and genuine non-travel days. Right: each purpose's share of all activities, including home. These are averages across people, not a claim that every person has this diary. Forced home-only bypasses are excluded. The common cohort contains the same recipients in every matching method.")

    for scope in (("common", "full") if manifest.get("primary_scope") == "common" else ("full", "common")):
        chains = tables["chains"].query("scope == @scope")
        ranking = chains.loc[chains.dataset.eq("Survey")].sort_values(["share", "chain"], ascending=[False, True]).head(15).chain.tolist()
        ranking_label = "survey"
        if not ranking:
            ranking = chains.groupby("chain").share.sum().sort_values(ascending=False).head(15).index.tolist()
            ranking_label = "population (no local survey)"
        fig, ax = plt.subplots(figsize=(12, 8))
        positions = np.arange(len(ranking) + 1)
        width = min(.24, .8 / len(datasets))
        for i, dataset in enumerate(datasets):
            values = chains.loc[chains.dataset.eq(dataset)].set_index("chain").share
            shown = values.reindex(ranking, fill_value=0).to_numpy()
            values_plot = np.r_[shown, max(0, 1 - shown.sum()) if len(values) else 0]
            ax.barh(positions + (i - (len(datasets) - 1) / 2) * width, values_plot * 100, height=width * .95, label=dataset, color=colors[dataset])
        ax.set_yticks(positions, ranking + ["All remaining chains"])
        ax.invert_yaxis()
        ax.set_xlabel("Share of people (%)")
        ax.set_title(f"15 most frequent {ranking_label} activity chains — {manifest.get('region', 'Switzerland')} — {scope} population")
        ax.legend()
        save(fig, f"{scope}_chains", "Chains describe the sequence of daily purposes. The same 15 chains, ranked by local survey-weighted frequency when available, are shown for all datasets; without a local survey they are ranked by pooled population shares. The last row contains the rest. Genuine home-only survey diaries remain in the denominator.")
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig.suptitle(f"Daily activity counts — {manifest.get('region', 'Switzerland')} — {scope} population")
        for ax, purpose in zip(axes.flat, PURPOSES):
            for dataset in datasets:
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
                for dataset in datasets:
                    rows = tables["activity_times"].query("scope == @scope and purpose == @purpose and metric == @metric and dataset == @dataset")
                    ax.step((rows.left_h + rows.right_h) / 2, rows.density, where="mid", label=dataset, color=colors[dataset])
                ax.set(title=LABELS[purpose], xlabel=f"{label} (hours)", ylabel="Density (per hour)", xlim=(0, 30))
            axes.flat[-1].axis("off")
            axes.flat[0].legend()
            save(fig, f"{scope}_{metric}", f"{label}: curves use half-hour bins and integrate to one when events are available. Starts exclude the artificial start of the initial activity; ends exclude the unknown end of the final activity. Durations include only completed internal activities. Missing, negative and out-of-range observations are excluded and counted in the CSV tables; empty curves mean no valid observations.")
    names = ", ".join(manifest.get("algorithms", ["V1", "V3"]))
    intro = (f"This report compares the diaries assigned by {names} with scenario-day Microcensus diaries. Survey results use person weights; population results weight each donor by the number of recipients assigned to that donor. Counts therefore retain non-travellers without expanding the full population activity table. "
        "The full-population figures use every age-eligible person retained by each method, including explicitly home-only residents. Methods can remove different people: coverage is shown below. The common-population figures repeat the comparison on exactly the same retained recipients, separating diary differences from coverage differences. "
        "These are assigned-diary distributions before downstream adjustments, not final simulated activity locations or observed population truth. Similarity to the survey alone does not establish which method predicts behaviour better.")
    if manifest.get("forced_home_only_excluded"):
        intro = intro.replace("including explicitly home-only residents", "excluding residents explicitly forced to stay home and any no-donor assignments")
        intro += " Genuine home-only survey diaries remain valid observations. The common cohort is the primary comparison and contains the same people for every algorithm."
    if manifest.get("execution_mode") == "debug":
        intro += " This analysis uses Synpp debug mode: population and survey input caches are reused without normal cache validation. Run-manifest cache identities and producer reports are saved alongside the results."
    if "V2" in manifest.get("algorithms", []):
        intro += " V1 uses ordered categorical matching; V2 learns a behavioural embedding from survey diaries; V3 compares category disagreements under hard constraints. These are descriptive population comparisons, not held-out predictive accuracy tests."
    body = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(names)} diary comparison</title>",
        "<style>body{max-width:1250px;margin:35px auto;font:17px/1.5 sans-serif;padding:15px}img{width:100%}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ccc}</style>",
        f"<h1>{html.escape(names)} population diary comparison: {html.escape(manifest.get('region', 'Switzerland'))}</h1>", f"<p>{html.escape(intro)}</p>",
        f"<p>Scenario day: {html.escape(str(manifest['scenario_day']))}. Survey respondents: {manifest['survey_respondents']:,}.</p>",
        "<h2>Coverage</h2>", tables["coverage"].to_html(index=False),
        "<h2>Activity rates</h2><p>Mean activities per resident per day, rates per 100 residents, percentage of residents doing each purpose, and activity composition. Two composition measures are shown: shares among non-home purposes, and shares among all activities including home. Each sums to 100% over its stated denominator.</p>",
        tables["purpose_rates"].rename(columns={"scope": "Cohort", "dataset": "Dataset", "purpose": "Purpose",
            "activities_per_100_people": "Activities / 100 people", "mean_activities_per_person": "Mean activities/person",
            "people_with_activity_percent": "Residents doing activity (%)", "share_of_nonhome_activities_percent": "Share of non-home activities (%)",
            "share_of_all_activities_percent": "Share of all activities (%)"}).to_html(index=False, na_rep="—", float_format=lambda x: f"{x:.2f}")]
    if regional_reports:
        navigation = " | ".join(f"<a href='{html.escape(r['directory'])}/report.html'>{html.escape(r['label'])}</a>" for r in regional_reports)
        body.insert(4, f"<p>Canton reports: {navigation}</p>")
    if manifest.get("canton_id") is not None:
        body.insert(4, "<p><a href='../report.html'>Switzerland and other cantons</a></p>"
            "<p>This report selects synthetic residents by their own home canton. Their assigned donors may live elsewhere. The survey comparison includes only respondents living in this canton. Each distribution is normalized within this canton and dataset; the top chains are ranked using this canton's survey weights. Empty curves mean no observations; zero survey respondents means no local survey reference is available.</p>")
    if (output / "interactive_distances.html").exists() and manifest.get("distance_maps", False):
        body += ["<h2>Interactive distance maps</h2><p>Swisstopo national map with a background-opacity control. Select a method, purpose and cohort. If online tiles fail, the grid and tables still work; static figures remain below.</p>",
                 "<p><a href='interactive_distances.html'>Open interactive map and cell table</a></p>",
                 "<iframe src='interactive_distances.html' title='Distance maps' style='width:100%;height:780px;border:1px solid #ccc'></iframe>"]
    if manifest.get("forced_home_only_excluded"):
        audit_path = "../cohort_exclusions.csv" if manifest.get("canton_id") is not None else "cohort_exclusions.csv"
        body += [f"<p>Excluded {manifest.get('excluded_people', 0):,} age-eligible recipients explicitly held out as home-only or lacking a real donor in any method. See <a href='{audit_path}'>regional exclusion counts</a>. Actual home-only survey diaries remain included.</p>"]
    if (output / "daily_activity_totals.csv").exists():
        daily = pd.read_csv(output / "daily_activity_totals.csv")
        body.insert(4, "<h2>Average resident's day</h2><p>Means per person per day, including zero-travel days. A genuine home-only day contributes one home activity. The home-only scenario bypass group is excluded in the three-method comparison.</p>" + daily.to_html(index=False, float_format=lambda x: f"{x:.2f}"))
    if manifest.get("density_analysis", False):
        density_coverage = pd.read_csv(output / "density_coverage.csv")
        body += ["<h2>Residential-density cohorts</h2><p>Survey strata use the respondent's own density; synthetic strata use the recipient's density. Population strata share the same matched recipients. Effective people summarises weight concentration only; it does not adjust for donor reuse. Density differences also reflect demographic differences and are not causal estimates.</p>",
                 density_coverage.to_html(index=False, na_rep="—", float_format=lambda x: f"{x:.1f}")]
    for name, explanation in figures:
        body += [f"<h2>{html.escape(name.replace('_', ' '))}</h2>", f"<p>{html.escape(explanation)}</p>", f"<img src='{name}.png' alt='{html.escape(name)}'>"]
    body.append("<p>All values are exported as CSV files alongside PNG/PDF figures. Re-render this report from these tables without rerunning matching using the population_comparison module.</p>")
    (output / "report.html").write_text("\n".join(body))
    return str(output / "report.html")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-render saved population comparison tables; no matching or simulation.")
    parser.add_argument("report_directory", type=Path)
    print(render_report(parser.parse_args().report_directory))
