"""Replay the municipal workplace controls with apprentices projected separately."""
from pathlib import Path

import pandas as pd

from analysis.apprentices.workplaces import CITIES, SCOPE
from synthesis.population.spatial.primary.work.assignment import MATRIX, WORK, assign, assignment_version, configure_assignment


def configure(context):
    # Replay both assignment passes locally, using the first pass to reserve capacity.
    configure_assignment(context, "fixed")
    context.stage(MATRIX["moving"])
    context.stage(SCOPE)
    context.stage("analysis.apprentices.workplaces")


def execute(context):

    # Select the survey-comparable non-apprentices and locate home-based workers.

    persons = context.stage("synthesis.population.sampled").set_index("person_id", verify_integrity=True)
    scope = context.stage(SCOPE).reindex(persons.index)
    if scope.isna().any():
        raise ValueError("Missing survey eligibility for sampled residents")
    eligible = scope & persons.employed.eq(context.stage("data.constants").EMPLOYED) & persons.job_position.ne(50)
    workers = persons.index[eligible]
    remote_ids = context.stage(WORK + "work_remotly").person_id
    home = persons.loc[remote_ids, "home_municipality_id"]
    employers = context.stage("data.statent.statent").set_index("enterprise_id", verify_integrity=True).municipality_id

    # Vary the controls while holding worker categories, OD inputs and apprentice stocks fixed.

    seed = context.config("random_seed")
    variants = [("ordinary", [], seed), ("ordinary_repeat", [], seed + 1),
                ("two_city", [6621, 5586], seed), ("two_city_repeat", [6621, 5586], seed + 1),
                ("three_city", CITIES, seed)]
    counts = {}
    for label, cities, trial_seed in variants:
        trial = Replay(context, cities, trial_seed)
        trial.fixed = assign(trial, "fixed", exclude_apprentices=True)
        moving = assign(trial, "moving", exclude_apprentices=True)
        workplaces = pd.concat([home, trial.fixed.set_index("person_id").destination_id.map(employers),
                                moving.set_index("person_id").destination_id.map(employers)], verify_integrity=True)
        locations = workplaces.reindex(workers)
        if locations.isna().any():
            raise ValueError(f"Missing workplace municipality for {locations.isna().sum()} workers in {label}")
        counts[label] = locations.astype(int).value_counts() / context.config("input_downsampling")

    # Compare each expanded scenario with both years of the reference survey.

    observed = context.stage("analysis.apprentices.workplaces").set_index(["year", "municipality"])
    counts = pd.DataFrame(counts).fillna(0)
    areas = observed.index.get_level_values("municipality").unique().union(counts.index)
    years = observed.index.get_level_values("year").unique()
    result = observed.reindex(pd.MultiIndex.from_product([years, areas], names=observed.index.names))
    stocks = ["recorded", "imputed", "reference", "respondents", "projected_apprentices"]
    result[stocks] = result[stocks].fillna(0)
    for label in counts:
        ordinary = result.index.get_level_values("municipality").map(counts[label]).fillna(0)
        result[label] = result.projected_apprentices + ordinary
    result = result.reset_index()
    result.to_csv(Path(context.path()) / "municipal_workplace_comparison.csv", index=False)
    return result


def validate(context):
    return assignment_version()


class Replay:
    """Use the requested city controls and reserve the current replay's fixed workplaces."""

    def __init__(self, context, cities, seed):
        self.context = context
        self.overrides = {"random_seed": seed, "workplace_calibration_municipalities": cities,
                          "workplace_unbiased_rounding": True, "workplace_reference_year": 2023}
        self.progress = context.progress
        self.fixed = None

    def config(self, name):
        return self.overrides[name] if name in self.overrides else self.context.config(name)

    def stage(self, name):
        if name == WORK + "fixed_work_locations":
            return self.fixed
        return self.context.stage(name)
