"""Single matching-report entry point: saved evidence by default, full tests on request."""

import hashlib
from pathlib import Path

from . import rebuild_reports
from synthesis.population.matching.V2.features import source_token


def configure(context):
    mode = context.config("matching_report_mode", "saved")
    context.config("analysis_path")
    context.config("random_seed")
    context.config("matching_validation_bootstrap", 500)
    context.config("matching_embedding_plot_method", "tsne")
    context.config("matching_embedding_plot_sample_size", 1500)
    if mode == "saved":
        context.config("matching_report_source")
    elif mode == "full":
        context.stage("analysis.matching.validation", alias="validation")
    else:
        raise ValueError("matching_report_mode must be saved or full")


def execute(context):

    # Select completed evidence or explicitly requested household-disjoint validation results.

    mode = context.config("matching_report_mode")
    source = (context.config("matching_report_source") if mode == "saved"
              else context.stage("validation"))
    output = Path(context.config("analysis_path")) / "matching" / "rebuilt"

    # Generate the comparison, embedding diagnostics and linked explanatory reports once.

    return rebuild_reports.build_reports(source, output,
        seed=int(context.config("random_seed")),
        bootstrap=int(context.config("matching_validation_bootstrap")),
        method=str(context.config("matching_embedding_plot_method")).lower(),
        maximum=int(context.config("matching_embedding_plot_sample_size")))


def validate(context):
    implementation = source_token(rebuild_reports)
    if context.config("matching_report_mode") == "saved":
        return hashlib.sha256((implementation + rebuild_reports.validate(context)).encode()).hexdigest()
    # Full mode's evidence is tracked by its declared validation dependency.
    from . import compare_matching, matching_embedding, explanations, geometry, metrics, reporting, report_diagnostics
    return hashlib.sha256((implementation + source_token(compare_matching, matching_embedding,
        explanations, geometry, metrics, reporting, report_diagnostics)).encode()).hexdigest()
