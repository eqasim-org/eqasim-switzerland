"""Pipeline adapter for the reusable V2 whole-diary matcher; V1 is independent."""

import logging
from pathlib import Path


from .V2 import configuration
from .V2.configuration import MatchingOptions
from .V2.diagnostics import MatchingDiagnostics
from .V2.inputs import prepare_inputs
from .V2.provenance import implementation_token
from .V2.wrappers import DiaryMatcher

logger = logging.getLogger("synpp")


def configure(context):
    configuration.configure(context)
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.activity_chains")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.primary.work.work_locations", alias="work_locations")
    context.stage("data.constants")
    context.stage("data.microcensus.commute")
    if context.config("census", "statpop") == "statpop":
        context.stage("synthesis.population.spatial.primary.education.locations", alias="education_locations")


def execute(context):

    # Copy inputs, enforce source eligibility and harmonize shared features.

    const = context.stage("data.constants")
    source, population, trips = prepare_inputs(context, const)
    eligible = (population["age"] >= const.MZ_AGE_THRESHOLD if const.census == "statpop"
                else population["age_class"] > 0)
    id_columns = ["person_id"] + (["household_id"] if const.census == "statpop" else [])
    result = population[id_columns].copy()
    result["mz_person_id"] = -1
    output = Path(context.path())
    output.mkdir(parents=True, exist_ok=True)
    if not eligible.any():
        return result, set()
    if source.empty:
        raise ValueError("No eligible domestic microcensus donors for the configured scenario day")

    # Fit or load a compatible model in this stage's own cache directory.

    target = population.loc[eligible].reset_index(drop=True)
    matcher = DiaryMatcher(MatchingOptions.from_context(context), cache_directory=output)
    matcher.fit(source, trips)

    # Retrieve learned neighbours within employment/student masks and sample complete diaries.

    prediction = matcher.predict(target)
    assigned = prediction.donor_ids
    matched = assigned >= 0
    result.loc[eligible, "mz_person_id"] = assigned
    removed = prediction.unmatched_person_ids

    # Export matching support, donor reuse and model feature availability for review.

    report = MatchingDiagnostics.write(output, matcher, population, target, prediction)
    if removed:
        logger.warning("V2 removed %d eligible persons without donors satisfying employment/student constraints; see matching_diagnostics.csv", len(removed))
    logger.info("V2 matched %d/%d eligible persons using %d donors", matched.sum(), len(target), report["used_donors"])

    # Preserve the downstream matching tuple and the original population order.

    result = result.loc[~result["person_id"].isin(removed)].reset_index(drop=True)
    if len(result) != len(population) - len(removed):
        raise RuntimeError("V2 matching changed population row alignment")
    return result, removed


def validate(context):
    return implementation_token()


# Older local callers used these private helpers. Keep forwarding only; all
# implementation and new consumers use V2's public classes/modules directly.
_prepare_inputs = prepare_inputs


def _fit_or_load(context, source, trips, output, seed):
    matcher = DiaryMatcher(MatchingOptions.from_context(context, random_seed=seed), cache_directory=output)
    matcher.fit(source, trips)
    return matcher.model_, matcher.processor_.scaler_, matcher.processor_.mappings_, matcher.training_report_
