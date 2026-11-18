"""V2 donor selection within the shared population matching workflow."""

import logging
from pathlib import Path


from .V2.configuration import MatchingOptions
from .V2.diagnostics import MatchingDiagnostics
from .V2.inputs import prepare_inputs
from .V2.provenance import implementation_token
from .V2.wrappers import DiaryMatcher
from .V2 import model as model_stage
from . import workflow

logger = logging.getLogger("synpp")


def configure(context):
    model_stage.configure(context)
    context.stage("synthesis.population.matching.V2.model")


def execute(context):

    # Copy inputs, enforce source eligibility and harmonize shared features.

    prepared = workflow.preprocess(context, "v2")
    source, population = prepared.source, prepared.population
    eligible = prepared.eligible
    result = workflow.initial_mapping(prepared)
    output = Path(context.path())
    output.mkdir(parents=True, exist_ok=True)
    if not eligible.any():
        return workflow.postprocess(prepared, result, set())
    if source.empty:
        raise ValueError("No eligible domestic microcensus donors for the configured scenario day")

    # Reuse the model fitted by the dedicated training stage.

    target = population.loc[eligible].reset_index(drop=True)
    matcher = context.stage("synthesis.population.matching.V2.model")

    # Retrieve learned neighbours within employment/student masks and sample complete diaries.

    prediction = matcher.match(target)
    assigned = prediction.donor_ids
    matched = assigned >= 0
    result.loc[eligible, "mz_person_id"] = assigned
    result, removed = workflow.postprocess(prepared, result, prediction.unmatched_person_ids)

    # Export matching support, donor reuse and model feature availability for review.

    report = MatchingDiagnostics.write(output, matcher, population, target, prediction, removed_ids=removed)
    if removed:
        logger.warning("V2 removed %d residents after unmatched-recipient and collective-household processing; see matching_diagnostics.csv", len(removed))
    logger.info("V2 matched %d/%d eligible persons using %d donors", matched.sum(), len(target), report["used_donors"])

    # Preserve the downstream matching tuple and the original population order.

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
