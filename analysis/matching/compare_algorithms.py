"""Compare V1, V2 and V3 on real matched recipients, nationally and by canton."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from synthesis.population.models.non_private import home_only_mask
from . import compare_v1_v3, population_comparison, distance_maps, density_comparison, interactive_maps


def configure(context):
    compare_v1_v3.configure(context)
    context.stage("synthesis.population.matching.matched_v2", alias="v2")
    context.config("matching_density_bounds", [400., 950., 1800.])


def execute(context):

    # Read the three assignments and remove recipients without a real donor consistently.

    mappings = {name.upper(): context.stage(name)[0].copy() for name in ("v1", "v2", "v3")}
    roster = context.stage("synthesis.population.sampled").copy()
    const = context.stage("data.constants")
    eligible = roster.age.ge(const.MZ_AGE_THRESHOLD) if const.census == "statpop" else roster.age_class.gt(0)
    held = home_only_mask(roster)
    eligible_ids = pd.Index(roster.loc[eligible, "person_id"])
    no_donor = set()
    for mapping in mappings.values():
        no_donor.update(mapping.loc[mapping.mz_person_id.lt(0) & mapping.person_id.isin(eligible_ids), "person_id"])
    excluded = set(roster.loc[held & eligible, "person_id"]) | no_donor
    keep = ~roster.person_id.isin(excluded)
    mappings = {name: frame.loc[~frame.person_id.isin(excluded)].copy() for name, frame in mappings.items()}
    output = Path(context.config("analysis_path")) / "matching" / "v1_v2_v3"
    output.mkdir(parents=True, exist_ok=True)
    audit = []
    for label, selector in [("Switzerland", eligible)] + [(name, eligible & roster.canton_id.eq(code)) for _, code, name in population_comparison.CANTONS]:
        audit.append(dict(region=label, input_eligible=int(selector.sum()),
            forced_home_only=int((selector & held).sum()), excluded_no_donor_any_method=int((selector & roster.person_id.isin(no_donor)).sum()),
            excluded_union=int((selector & ~keep).sum()), retained_before_matching_coverage=int((selector & keep).sum())))
    pd.DataFrame(audit).to_csv(output / "cohort_exclusions.csv", index=False)
    sources = {name: str(context.path(name)) for name in ("v1", "v2", "v3", "synthesis.population.sampled", "data.microcensus.persons", "data.microcensus.trips", "data.spatial.cantons")}
    reports = {}
    for name in ("v1", "v2", "v3"):
        path = Path(sources[name]) / "matching_report.json"
        if path.exists():
            reports[name.upper()] = json.loads(path.read_text())
    (output / "run_manifest.json").write_text(json.dumps({"input_cache_directories": sources,
        "producer_reports": reports, "mode": "debug" if type(context).__name__.startswith("Debug") else "normal"}, indent=2))

    # Generate descriptive distributions on common and full retained cohorts, plus density strata.

    report = population_comparison.generate(output, mappings, roster.loc[keep].copy(),
        context.stage("data.microcensus.persons"), context.stage("data.microcensus.trips")[0],
        context.stage("data.microcensus.activity_chains"), const, context.config("specific_day_scenario"),
        cantons=context.stage("data.spatial.cantons"), distance_map_options={
            "grid_size": context.config("matching_comparison_grid_size"),
            "canton_grid_size": context.config("matching_comparison_canton_grid_size"),
            "min_people": context.config("matching_comparison_map_min_people")},
        report_metadata={"forced_home_only_excluded": True, "excluded_people": len(excluded), "primary_scope": "common",
                         "exclusions_by_region": {row["region"]: row["excluded_union"] for row in audit},
                         "execution_mode": "debug" if type(context).__name__.startswith("Debug") else "normal"},
        density_bounds=context.config("matching_density_bounds"))
    return report


def validate(context):
    modules = (population_comparison, distance_maps, density_comparison, interactive_maps)
    return hashlib.sha256(b"".join(Path(module.__file__).read_bytes() for module in modules) +
        Path(interactive_maps.__file__).with_name("interactive_distances.html.template").read_bytes()).hexdigest()
