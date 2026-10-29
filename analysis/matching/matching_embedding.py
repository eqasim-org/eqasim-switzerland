"""Scientific representation diagnostics using outer-held-out respondents."""

import json
from pathlib import Path
import shutil

import pandas as pd

from .geometry import geometry_diagnostics, importance_figure, permutation_summary
from .reporting import write_html
from . import geometry, reporting
from synthesis.population.matching.V2.features import source_token


def configure(context):
    context.config("analysis_path")
    context.config("random_seed")
    context.config("matching_embedding_plot_method", "tsne")
    context.config("matching_embedding_plot_sample_size", 1500)
    context.config("matching_validation_bootstrap", 500)
    context.stage("analysis.matching.validation", alias="validation")


def execute(context):

    # Use fold-specific unseen embeddings so training fit cannot masquerade as representation quality.

    source = Path(context.stage("validation"))
    output = Path(context.config("analysis_path")) / "matching" / "embedding"
    return generate_report(source, output, int(context.config("random_seed")),
                           int(context.config("matching_validation_bootstrap")),
                           str(context.config("matching_embedding_plot_method")).lower(),
                           int(context.config("matching_embedding_plot_sample_size")))


def generate_report(source, output, seed, bootstrap, method="tsne", maximum=1500):
    """Render saved representations and permutation deltas, without training or matching."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())
    if method not in ("pca", "tsne"):
        raise ValueError("matching_embedding_plot_method must be pca or tsne")
    if maximum < 5:
        raise ValueError("matching_embedding_plot_sample_size must be at least five")

    # Measure neighborhood usefulness, projection fidelity, model stability and predictive contributions.

    diagnostics, stability = geometry_diagnostics(source, output, manifest, seed, maximum, method)
    diagnostics.to_csv(output / "geometry.csv", index=False)
    stability.to_csv(output / "seed_stability.csv", index=False)
    respondents = pd.read_csv(source / "respondents.csv")
    importance = permutation_summary(pd.read_csv(source / "grouped_permutation.csv"), respondents,
                                     bootstrap, seed)
    importance.to_csv(output / "grouped_predictive_importance.csv", index=False)
    importance_figure(importance, output)

    # Export vector figures with auditable data and explicit limits on scientific interpretation.

    for name in ("manifest.json", "feature_inventory.csv", "grouped_permutation.csv"):
        shutil.copy2(source / name, output / name)
    write_html(output, "Matching embeddings: predictive representation evidence",
        "The model represents each person's age, income, household and residential characteristics as 16 learned "
        "numbers. People nearby in that space are candidate diary donors. These figures ask whether the space "
        "organizes useful information about travel, how sensitive the picture is to its plotting settings, and "
        "which inputs helped the model predict hidden survey days. All figures reuse saved embeddings, predictions "
        "and permutation results; rebuilding this report does not fit a model or rematch anyone. "
        "All quantitative tests use outer-held-out respondents. Geometry is measured in the original latent space; "
        "2D projections are illustrations whose neighborhood trustworthiness is reported. Neighbour behaviour distances "
        "are descriptive; predictive scores in the comparison report are the main performance measure. Seed stability "
        "uses identical outer folds and compares neighbourhoods and distances, avoiding arbitrary latent axes. Grouped "
        "permutation preserves employment/student masks and joint household counts. Contributions are predictive "
        "associations, not causal effects or proof that a feature is irrelevant. Correlated feature groups can mask or "
        "redistribute importance. Clustered intervals average model/permutation repeats before resampling people.",
        [("Held-out neighbourhood and geometry diagnostics", diagnostics), ("Seed stability", stability),
         ("Projection sensitivity and neighbourhood fidelity", pd.read_csv(output / "projection_sensitivity.csv")),
         ("Grouped predictive contribution with 95% intervals", importance)],
        ["heldout_representation", "embedding_behaviour", "projection_sensitivity", "neighbour_behaviour",
         "predictive_feature_contributions"])
    return str(output / "report.html")


def validate(context):
    from . import explanations
    return source_token(geometry, reporting, explanations)
