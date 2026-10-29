"""Compare individual predictive distributions on genuinely unseen diaries."""

import json
from pathlib import Path
import shutil

import pandas as pd

from .reporting import (calibration_tables, comparison_figures, conditional_summary,
                        paired_summary, write_html)
from . import reporting
from . import report_diagnostics
from synthesis.population.matching.V2.features import source_token


def configure(context):
    context.config("analysis_path")
    context.config("random_seed")
    context.config("matching_validation_bootstrap", 500)
    context.stage("analysis.matching.validation", alias="validation")


def execute(context):

    # Load cross-fitted predictions; neither final population matches nor national totals are targets.

    source = Path(context.stage("validation"))
    output = Path(context.config("analysis_path")) / "matching" / "comparison"
    return generate_report(source, output, int(context.config("random_seed")),
                           int(context.config("matching_validation_bootstrap")))


def generate_report(source, output, seed, bootstrap):
    """Render saved predictions without importing or invoking the validation stage."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(source / "out_of_fold_scores.csv")
    respondents = pd.read_csv(source / "respondents.csv")
    manifest = json.loads((source / "manifest.json").read_text())

    # Estimate paired predictive skill and household-clustered confidence intervals.

    summary, differences, average = paired_summary(
        scores, respondents, bootstrap, seed)
    summary.to_csv(output / "predictive_scores.csv", index=False)
    differences.to_csv(output / "paired_differences.csv", index=False)
    conditional_summary(average).to_csv(output / "conditional_risk.csv", index=False)
    reliability, ranks = calibration_tables(scores, respondents, manifest["draws"], seed,
                                             bootstrap)
    reliability.to_csv(output / "reliability.csv", index=False)
    ranks.to_csv(output / "randomized_ranks.csv", index=False)
    coverage = scores.groupby("algorithm", as_index=False)["covered"].mean()
    comparison_figures(output, differences, reliability, ranks, average)
    extra_figures, extra_tables = report_diagnostics.comparison_diagnostics(
        output, scores, respondents, average, bootstrap, seed)

    # Publish scientific figures alongside auditable evidence and interpretation limits.

    for name in ("manifest.json", "feature_inventory.csv", "population_support.csv", "covariate_balance.csv", "respondents.csv", "out_of_fold_scores.csv"):
        shutil.copy2(source / name, output / name)
    write_html(output, "Matching: held-out predictive transfer",
        f"We tested matching using {manifest['respondents']:,} real survey respondents. We hid one group of households "
        "at a time, learned V2 using the remaining households, and asked V1 and V2 to assign days to the hidden people "
        f"using only their personal and household characteristics. Each person received {manifest['draws']} sampled donor days "
        "per algorithm and model repeat. Their own diary and household were excluded from the donor pool. We then checked "
        "how well those possible days described the real day we had hidden. This is survey-to-survey prediction without "
        "self-matching. The figures below reuse those saved tests; no models were fitted for this report. "
        "Lower error scores are better. "
        "Intervals resample households after averaging model repeats. Survey-weighted risk and covariate-shift target risk "
        "are separate estimands. Neither proves that an unobserved population's travel behaviour is correct. "
        "National activity-chain shares are not an accuracy target. Check coverage before paired differences: paired skill "
        "uses respondents covered by both algorithms. Calibration plots are descriptive and include finite-ensemble/discrete effects.",
        [("Coverage: fraction receiving a complete donor ensemble", coverage),
         ("Paired differences with 95% household intervals", differences),
         ("Source/target overlap", pd.DataFrame([manifest["shift"]]))] + extra_tables,
        ["paired_predictive_skill"] + extra_figures + [
         "event_reliability", "rank_calibration", "continuous_age_skill"],
        finding=report_diagnostics.saved_findings(differences))
    return str(output / "report.html")


def validate(context):
    from . import explanations
    return source_token(reporting, report_diagnostics, explanations)
