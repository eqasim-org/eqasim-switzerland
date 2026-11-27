"""Paired V1/V2/V3 comparison of density-only and activity-zone diary matching."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import shapely

from data.microcensus.activity_chain_features import (
    FEATURES,
    daily_trip_counts,
    effective_sample,
    valid_diary_mask,
    implementation_token,
)
from data.utils import coerce_boolean_series
from synthesis.population.matching import (
    matched_v1,
    matched_v2,
    matched_v3,
    spatial_feature,
    workflow,
)
from synthesis.population.matching.V2 import model as v2_model
from synthesis.population.matching.V2.configuration import MatchingOptions
from synthesis.population.models.non_private import home_only_mask

VARIANTS = (
    ("v1_density", matched_v1, "density"),
    ("v1_zones", matched_v1, "activity_chain_cluster"),
    ("v2_density", matched_v2, "density"),
    ("v2_zones", matched_v2, "activity_chain_cluster"),
    ("v3_density", matched_v3, "density"),
    ("v3_zones", matched_v3, "activity_chain_cluster"),
)


def configure(context):
    matched_v3.configure(context)
    v2_model.configure(context)
    context.stage(spatial_feature.STAGE)
    context.stage("data.spatial.cantons")
    context.stage("data.spatial.municipalities")
    context.config("analysis_path")
    context.config("data_path")
    context.config("input_downsampling")


def execute(context):

    # Fix the original population, region definitions and experiment identity.

    roster = context.stage("synthesis.population.sampled")
    zones = context.stage(spatial_feature.STAGE)
    const = context.stage("data.constants")
    if const.census != "statpop":
        raise ValueError("The paired zone experiment requires STATPOP")
    identity = {
        "seed": int(context.config("random_seed")),
        "day": context.config("specific_day_scenario"),
        "sample": float(context.config("input_downsampling")),
        "population": len(roster),
        "opening_hours": bool(context.config("consider_opening_hours")),
        "minimum_observations": int(context.config("matching_minimum_observations")),
        "density_bounds": list(context.config("matching_density_bounds")),
        "implementation": validate(context),
        "zones": zones["metadata"],
        "zoning_stage": spatial_feature.STAGE,
        "v2_options": repr(MatchingOptions.from_context(context)),
    }
    for label, frame in [
        ("population", roster),
        ("survey", context.stage("data.microcensus.persons")),
    ]:
        columns = [
            c
            for c in [
                "person_id",
                "age",
                "home_x",
                "home_y",
                "employed",
                "car_availability",
                "employment_status",
                "ovgk",
                "population_density",
                "income_class",
                "is_student",
                "non_private_home_only_households",
                "collective_housing_resident",
            ]
            if c in frame
        ]
        identity[label + "_sha256"] = hashlib.sha256(
            pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().tobytes()
        ).hexdigest()
    identity["zone_assignments_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(
            zones["municipalities"][["municipality_id", "activity_chain_cluster"]],
            index=False,
        )
        .to_numpy()
        .tobytes()
    ).hexdigest()
    identity["zone_geometry_sha256"] = hashlib.sha256(
        b"".join(zones["municipalities"].geometry.to_wkb())
    ).hexdigest()
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    output = (
        Path(context.config("analysis_path"))
        / "matching"
        / ("activity_chain_zones_" + digest)
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Existing experiment identity differs")
    manifest.write_text(json.dumps(identity, indent=2), encoding="utf-8")

    # Run the production matchers with paired inputs and independent V2 fits.
    # V1/V3 replace density; V2 adds zones while retaining every density input.

    mappings, coverage = [], []
    for name, module, feature in VARIANTS:
        directory = output / name
        directory.mkdir(exist_ok=True)
        path = directory / "mapping.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            seconds = json.loads((directory / "run.json").read_text())["seconds"]
            removed = pd.read_csv(directory / "removed.csv").person_id.tolist()
            print(f"{name}: reused completed checkpoint", flush=True)
        else:
            print(f"{name}: starting {len(roster):,} residents", flush=True)
            started = time.perf_counter()
            proxy = VariantContext(context, directory, feature)
            frame, removed = module.execute(proxy)
            seconds = time.perf_counter() - started
            pd.DataFrame({"person_id": sorted(removed)}).to_csv(
                directory / "removed.csv", index=False
            )
            (directory / "run.json").write_text(
                json.dumps(
                    {
                        "seconds": seconds,
                        "spatial_feature": feature,
                        "removed": len(removed),
                        "retained": len(frame),
                    }
                ),
                encoding="utf-8",
            )
            frame.to_parquet(path, index=False)
            print(
                f"{name}: finished in {seconds:.1f}s; retained {len(frame):,}; removed {len(removed):,}",
                flush=True,
            )
        mappings.append((name, frame))
        coverage.append(
            {
                "variant": name,
                "input_persons": len(roster),
                "retained": len(frame),
                "removed": len(removed),
                "seconds": seconds,
            }
        )
        pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)

    # Compare an identical, non-forced-home cohort across all six assignments.

    survey = workflow.survey_for_day(context)
    counts = daily_trip_counts(context.stage("data.microcensus.trips")[0])
    valid = valid_diary_mask(context.stage("data.microcensus.activity_chains"))
    survey = survey.loc[survey.person_id.map(valid).fillna(False)].copy()
    analysis_mask, exclusions = analysis_cohort(roster, const)
    selected = pd.Index(roster.loc[analysis_mask, "person_id"])
    for name, frame in mappings:
        selected = selected.intersection(
            frame.loc[
                frame.mz_person_id.ge(0) & frame.mz_person_id.map(valid).fillna(False),
                "person_id",
            ],
            sort=False,
        )
    cohort = roster.loc[roster.person_id.isin(selected)].copy()
    exclusions.append({"reason": "common_valid_matched_cohort", "persons": len(cohort)})
    pd.DataFrame(exclusions).to_csv(output / "exclusions.csv", index=False)
    rows = regional_metrics(
        cohort,
        survey,
        dict(mappings),
        counts,
        context.stage("data.spatial.cantons"),
        context.stage("data.spatial.municipalities")[0],
        float(context.config("input_downsampling")),
    )
    rows.to_csv(output / "regional_trip_counts.csv", index=False)
    changes = improvement_table(rows)
    changes.to_csv(output / "improvements.csv", index=False)
    protocol = {
        "common_recipients": len(cohort),
        "survey_reference_persons": len(survey),
        "excluded_from_analysis": len(roster) - len(cohort),
        "forced_home_definition": "Explicit home-only bypasses, collective home-only donor restrictions, and children below MZ age threshold. Optional truck-driver forced-home flag when present.",
        "cohort": "Same residents with valid non-missing donor diaries in all six variants; natural zero-trip days retained.",
        "zoning_stage": spatial_feature.STAGE,
        "zone_definition": "Whole municipalities, resident-plurality assignment from four-feature activity contours; settings are recorded in manifest zones.source_metadata.",
        "v2_design": "Zones are an additional categorical geographic input; Existing density inputs remain in both V2 variants. Models are fitted separately with identical seeds and split rules.",
        "geography": "Residence canton or municipality; Switzerland aggregate and 26 cantons plus Lausanne, Geneva and Zurich municipalities.",
        "counts": "Trips by destination purpose, secondary work included; all return-home trips counted in total.",
        "reference_counts": "Weighted MZ mean trips/person multiplied by common-cohort regional population; full scale divides by input_downsampling.",
        "interpretation": "Descriptive in-sample fit to MZ 2015. Zones were learned on this same survey; not independent predictive validation, official observed canton totals, or simulated/rerouted MATSim trips.",
        "uncertainty": "One paired random seed per algorithm. Small effective MZ samples flagged; differences have no significance claim.",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    # Produce regional comparisons and a standalone illustrated report.

    from .activity_chain_zones_report import render

    print("ANALYSIS", output, flush=True)
    return str(render(output))


class VariantContext:
    def __init__(self, parent, path, feature):
        self.parent, self.directory, self.feature = parent, Path(path), feature
        self.model = None

    def stage(self, name):
        if name == "synthesis.population.matching.V2.model":
            if self.model is None:
                self.model = v2_model.execute(self)
            return self.model
        return self.parent.stage(name)

    def config(self, name):
        return (
            self.feature if name == spatial_feature.OPTION else self.parent.config(name)
        )

    def path(self, *args):
        return str(self.directory) if not args else self.parent.path(*args)

    def __getattr__(self, name):
        return getattr(self.parent, name)


def analysis_cohort(roster, const):
    below = roster.age.lt(const.MZ_AGE_THRESHOLD)
    held = home_only_mask(roster)
    collective = coerce_boolean_series(
        roster.get("collective_housing_resident", pd.Series(False, index=roster.index)),
        name="collective_housing_resident",
    )
    truck = coerce_boolean_series(
        roster.get("is_truck_driver", pd.Series(False, index=roster.index)),
        name="is_truck_driver",
    )
    exclusions = [
        {"reason": name, "persons": int(mask.sum())}
        for name, mask in [
            ("below_survey_age", below),
            ("explicit_home_only", held),
            ("collective_home_only_donors", collective),
            ("truck_drivers_if_flagged", truck),
        ]
    ]
    keep = ~(below | held | collective | truck)
    exclusions.append(
        {"reason": "eligible_before_common_match_filter", "persons": int(keep.sum())}
    )
    return keep, exclusions


def regional_metrics(cohort, survey, mappings, counts, cantons, municipalities, sample):
    if not np.isfinite(sample) or not 0 < sample <= 1:
        raise ValueError("Population sampling fraction must be in (0,1]")
    if cohort.person_id.duplicated().any() or survey.person_id.duplicated().any():
        raise ValueError("Unique person IDs required for regional comparison")
    observed = counts.reindex(survey.person_id, fill_value=0).to_numpy(dtype=float)
    weights = survey.person_weight.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Reference survey weights must be positive and finite")
    predicted = {}
    for name, frame in mappings.items():
        donors = cohort.person_id.map(frame.set_index("person_id").mz_person_id)
        if donors.isna().any() or donors.lt(0).any():
            raise ValueError("Common cohort contains missing assignments")
        predicted[name] = counts.reindex(donors, fill_value=0).to_numpy(dtype=float)
    regions = [
        (
            "national",
            "Switzerland",
            0,
            np.ones(len(cohort), bool),
            np.ones(len(survey), bool),
        )
    ]
    for row in (
        cantons.loc[cantons.canton_id.between(1, 26)]
        .sort_values("canton_id")
        .itertuples()
    ):
        name = getattr(row, "canton_name_en", row.canton_name)
        regions.append(
            (
                "canton",
                name,
                int(row.canton_id),
                cohort.canton_id.eq(row.canton_id).to_numpy(),
                survey.canton_id.eq(row.canton_id).to_numpy(),
            )
        )
    for mid, name in [(5586, "Lausanne"), (6621, "Geneva"), (261, "Zurich")]:
        polygon = municipalities.loc[
            municipalities.municipality_id.eq(mid)
        ].geometry.union_all()
        regions.append(
            (
                "municipality",
                name,
                mid,
                shapely.intersects_xy(
                    polygon, cohort.home_x.to_numpy(), cohort.home_y.to_numpy()
                ),
                shapely.intersects_xy(
                    polygon, survey.home_x.to_numpy(), survey.home_y.to_numpy()
                ),
            )
        )
    rows = []
    for level, region, code, target_mask, source_mask in regions:
        n = int(target_mask.sum())
        n_survey = int(source_mask.sum())
        if n_survey:
            reference = np.average(
                observed[source_mask], axis=0, weights=weights[source_mask]
            )
            neff = effective_sample(weights[source_mask])
        else:
            reference = np.full(len(FEATURES), np.nan)
            neff = 0.0
        for variant, values in predicted.items():
            totals = values[target_mask].sum(axis=0)
            rates = totals / n if n else np.full(len(FEATURES), np.nan)
            for index, feature in enumerate(FEATURES):
                expected = n * reference[index]
                rows.append(
                    {
                        "level": level,
                        "region": region,
                        "region_id": code,
                        "variant": variant,
                        "feature": feature,
                        "population_persons": n,
                        "survey_persons": n_survey,
                        "survey_effective_sample": neff,
                        "low_survey_support": neff < 100,
                        "survey_mean": reference[index],
                        "matched_mean": rates[index],
                        "matched_trips_sample": totals[index],
                        "expected_trips_sample": expected,
                        "matched_trips_full_scale": totals[index] / sample,
                        "expected_trips_full_scale": expected / sample,
                        "relative_error_percent": 100
                        * (rates[index] - reference[index])
                        / reference[index]
                        if reference[index] > 0
                        else np.nan,
                        "absolute_rate_error": abs(rates[index] - reference[index]),
                    }
                )
    return pd.DataFrame(rows)


def improvement_table(metrics):
    keys = [
        "level",
        "region",
        "region_id",
        "feature",
        "population_persons",
        "survey_effective_sample",
        "low_survey_support",
    ]
    output = []
    for algorithm in ["v1", "v2", "v3"]:
        baseline = metrics.loc[metrics.variant.eq(algorithm + "_density")].set_index(
            keys
        )
        zones = metrics.loc[metrics.variant.eq(algorithm + "_zones")].set_index(keys)
        table = baseline[
            [
                "survey_mean",
                "matched_mean",
                "relative_error_percent",
                "absolute_rate_error",
            ]
        ].rename(
            columns={
                c: "density_" + c
                for c in [
                    "matched_mean",
                    "relative_error_percent",
                    "absolute_rate_error",
                ]
            }
        )
        for column in ["matched_mean", "relative_error_percent", "absolute_rate_error"]:
            table["zones_" + column] = zones[column]
        table["absolute_error_reduction"] = (
            table.density_absolute_rate_error - table.zones_absolute_rate_error
        )
        table["zones_better"] = table.absolute_error_reduction.gt(0)
        table["algorithm"] = algorithm.upper()
        output.append(table.reset_index())
    return pd.concat(output, ignore_index=True)


def validate(context):
    modules = [matched_v1, matched_v2, matched_v3, workflow, spatial_feature]
    paths = [Path(m.__file__) for m in modules] + [Path(__file__)]
    payload = b"".join(p.read_bytes() for p in paths)
    payload += (
        matched_v1.validate(context)
        + matched_v2.validate(context)
        + matched_v3.validate(context)
        + implementation_token()
    ).encode()
    return hashlib.sha256(payload).hexdigest()
